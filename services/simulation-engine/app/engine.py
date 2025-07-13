#!/usr/bin/env python3
"""
FinSim Simulation Engine

Deterministic, event-driven order-book matching with queue-priority microstructure.
Implements continuous-time double-auction with price-time priority.
"""

import asyncio
import json
import logging
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

# Data Models
@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    timestamp: datetime = None
    agent_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def __lt__(self, other):
        """For heapq priority - price-time priority"""
        if self.side == OrderSide.BUY:
            # For buy orders, higher price has higher priority
            if self.price != other.price:
                return self.price > other.price
        else:
            # For sell orders, lower price has higher priority
            if self.price != other.price:
                return self.price < other.price
        
        # If prices are equal, earlier timestamp has priority
        return self.timestamp < other.timestamp

@dataclass
class Trade:
    trade_id: str
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    agent_id: str

class OrderBookSnapshot(BaseModel):
    symbol: str
    bids: List[Tuple[float, int]]  # (price, quantity)
    asks: List[Tuple[float, int]]  # (price, quantity)
    last_trade_price: Optional[float] = None
    timestamp: datetime

# Order Book Implementation
class OrderBook:
    """Continuous-time double-auction order book with price-time priority"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: List[Order] = []  # Max heap for buy orders
        self.asks: List[Order] = []  # Min heap for sell orders
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.last_trade_price: Optional[float] = None
        
    def add_order(self, order: Order) -> List[Trade]:
        """Add order to book and return any resulting trades"""
        self.orders[order.order_id] = order
        trades = []
        
        if order.order_type == OrderType.MARKET:
            trades = self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._execute_limit_order(order)
        
        return trades
    
    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against best available prices"""
        trades = []
        remaining_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            # Match against best ask prices
            while remaining_qty > 0 and self.asks:
                best_ask = heapq.heappop(self.asks)
                trade_qty = min(remaining_qty, best_ask.quantity - best_ask.filled_quantity)
                
                trade = self._create_trade(order, best_ask, trade_qty, best_ask.price)
                trades.append(trade)
                
                remaining_qty -= trade_qty
                best_ask.filled_quantity += trade_qty
                
                if best_ask.filled_quantity < best_ask.quantity:
                    # Partial fill, put back in book
                    heapq.heappush(self.asks, best_ask)
                    best_ask.status = OrderStatus.PARTIAL
                else:
                    best_ask.status = OrderStatus.FILLED
        
        else:  # SELL
            # Match against best bid prices
            while remaining_qty > 0 and self.bids:
                best_bid = heapq.heappop(self.bids)
                trade_qty = min(remaining_qty, best_bid.quantity - best_bid.filled_quantity)
                
                trade = self._create_trade(best_bid, order, trade_qty, best_bid.price)
                trades.append(trade)
                
                remaining_qty -= trade_qty
                best_bid.filled_quantity += trade_qty
                
                if best_bid.filled_quantity < best_bid.quantity:
                    # Partial fill, put back in book
                    heapq.heappush(self.bids, best_bid)
                    best_bid.status = OrderStatus.PARTIAL
                else:
                    best_bid.status = OrderStatus.FILLED
        
        # Update order status
        order.filled_quantity = order.quantity - remaining_qty
        if order.filled_quantity == order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL
        
        return trades
    
    def _execute_limit_order(self, order: Order) -> List[Trade]:
        """Execute limit order with price-time priority matching"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # Check if we can match against asks
            while self.asks and self.asks[0].price <= order.price and order.filled_quantity < order.quantity:
                best_ask = heapq.heappop(self.asks)
                trade_qty = min(
                    order.quantity - order.filled_quantity,
                    best_ask.quantity - best_ask.filled_quantity
                )
                
                # Execute at mid-quote minus micro-tick for price improvement
                trade_price = (order.price + best_ask.price) / 2 - 0.01
                trade = self._create_trade(order, best_ask, trade_qty, trade_price)
                trades.append(trade)
                
                order.filled_quantity += trade_qty
                best_ask.filled_quantity += trade_qty
                
                if best_ask.filled_quantity < best_ask.quantity:
                    heapq.heappush(self.asks, best_ask)
                    best_ask.status = OrderStatus.PARTIAL
                else:
                    best_ask.status = OrderStatus.FILLED
            
            # If not fully filled, add to book
            if order.filled_quantity < order.quantity:
                heapq.heappush(self.bids, order)
                if order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.FILLED
        
        else:  # SELL
            # Check if we can match against bids
            while self.bids and self.bids[0].price >= order.price and order.filled_quantity < order.quantity:
                best_bid = heapq.heappop(self.bids)
                trade_qty = min(
                    order.quantity - order.filled_quantity,
                    best_bid.quantity - best_bid.filled_quantity
                )
                
                # Execute at mid-quote minus micro-tick for price improvement
                trade_price = (best_bid.price + order.price) / 2 - 0.01
                trade = self._create_trade(best_bid, order, trade_qty, trade_price)
                trades.append(trade)
                
                order.filled_quantity += trade_qty
                best_bid.filled_quantity += trade_qty
                
                if best_bid.filled_quantity < best_bid.quantity:
                    heapq.heappush(self.bids, best_bid)
                    best_bid.status = OrderStatus.PARTIAL
                else:
                    best_bid.status = OrderStatus.FILLED
            
            # If not fully filled, add to book
            if order.filled_quantity < order.quantity:
                heapq.heappush(self.asks, order)
                if order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.FILLED
        
        return trades
    
    def _create_trade(self, buy_order: Order, sell_order: Order, quantity: int, price: float) -> Trade:
        """Create a trade between two orders"""
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id
        )
        
        self.trades.append(trade)
        self.last_trade_price = price
        return trade
    
    def get_snapshot(self) -> OrderBookSnapshot:
        """Get current order book snapshot"""
        # Aggregate bids and asks by price level
        bid_levels = {}
        ask_levels = {}
        
        for order in self.bids:
            remaining_qty = order.quantity - order.filled_quantity
            if remaining_qty > 0:
                bid_levels[order.price] = bid_levels.get(order.price, 0) + remaining_qty
        
        for order in self.asks:
            remaining_qty = order.quantity - order.filled_quantity
            if remaining_qty > 0:
                ask_levels[order.price] = ask_levels.get(order.price, 0) + remaining_qty
        
        # Sort bids (highest first) and asks (lowest first)
        bids = sorted(bid_levels.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(ask_levels.items(), key=lambda x: x[0])
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            bids=bids[:10],  # Top 10 levels
            asks=asks[:10],  # Top 10 levels
            last_trade_price=self.last_trade_price,
            timestamp=datetime.now()
        )

# Global variables
order_books: Dict[str, OrderBook] = {}
kafka_producer = None
redis_client = None
websocket_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global kafka_producer, redis_client
    
    # Startup
    logger.info("Starting Simulation Engine...")
    
    # Initialize Kafka producer
    kafka_producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://redis:6379")
    
    # Initialize order books for common symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
    for symbol in symbols:
        order_books[symbol] = OrderBook(symbol)
    
    logger.info("Simulation Engine started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Simulation Engine...")
    if kafka_producer:
        kafka_producer.close()
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="FinSim Simulation Engine",
    description="Event-driven order-book matching engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/api/v1/orders")
async def place_order(order_request: OrderRequest):
    """Place a new order"""
    try:
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            agent_id=order_request.agent_id
        )
        
        # Get or create order book
        if order.symbol not in order_books:
            order_books[order.symbol] = OrderBook(order.symbol)
        
        order_book = order_books[order.symbol]
        
        # Execute order
        trades = order_book.add_order(order)
        
        # Publish trades to Kafka
        for trade in trades:
            if kafka_producer:
                kafka_producer.send('trades', value=asdict(trade))
        
        # Publish order book update
        snapshot = order_book.get_snapshot()
        if kafka_producer:
            kafka_producer.send('orderbook', value=snapshot.model_dump())
        
        # Send updates to WebSocket clients
        for websocket in websocket_connections[:]:
            try:
                await websocket.send_json({
                    "type": "order_update",
                    "order": asdict(order),
                    "trades": [asdict(t) for t in trades],
                    "orderbook": snapshot.model_dump()
                })
            except:
                websocket_connections.remove(websocket)
        
        return {
            "order": asdict(order),
            "trades": [asdict(t) for t in trades],
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    """Get order book snapshot for a symbol"""
    try:
        if symbol not in order_books:
            raise HTTPException(status_code=404, detail=f"Order book not found for symbol {symbol}")
        
        order_book = order_books[symbol]
        snapshot = order_book.get_snapshot()
        
        return snapshot.model_dump()
    
    except Exception as e:
        logger.error(f"Error getting order book for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trades/{symbol}")
async def get_trades(symbol: str, limit: int = 100):
    """Get recent trades for a symbol"""
    try:
        if symbol not in order_books:
            raise HTTPException(status_code=404, detail=f"Order book not found for symbol {symbol}")
        
        order_book = order_books[symbol]
        recent_trades = order_book.trades[-limit:]
        
        return [asdict(trade) for trade in recent_trades]
    
    except Exception as e:
        logger.error(f"Error getting trades for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/orders/{order_id}")
async def get_order(order_id: str):
    """Get order details"""
    try:
        # Search for order across all order books
        for order_book in order_books.values():
            if order_id in order_book.orders:
                order = order_book.orders[order_id]
                return asdict(order)
        
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
    
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    try:
        # Search for order across all order books
        for order_book in order_books.values():
            if order_id in order_book.orders:
                order = order_book.orders[order_id]
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                    order.status = OrderStatus.CANCELLED
                    
                    # Remove from order book heaps
                    if order.side == OrderSide.BUY:
                        order_book.bids = [o for o in order_book.bids if o.order_id != order_id]
                        heapq.heapify(order_book.bids)
                    else:
                        order_book.asks = [o for o in order_book.asks if o.order_id != order_id]
                        heapq.heapify(order_book.asks)
                    
                    return {"status": "cancelled", "order": asdict(order)}
                else:
                    raise HTTPException(status_code=400, detail="Order cannot be cancelled")
        
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
    
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "engine:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )