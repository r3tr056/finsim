#!/usr/bin/env python3
"""
FinSim Market Data Service

FastAPI service that streams real-time and historical market data into Kafka topics.
Supports multiple data providers: Bloomberg API, Refinitiv Eikon, Yahoo Finance, Quandl, Polygon.io
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
from kafka import KafkaProducer
import pandas as pd
import numpy as np
import yfinance as yf
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class MarketDataRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"

class QuoteData(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
    exchange: str = "NYSE"

class FundamentalData(BaseModel):
    symbol: str
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    revenue: Optional[float]
    net_income: Optional[float]
    timestamp: datetime

# Global variables
kafka_producer = None
redis_client = None
websocket_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global kafka_producer, redis_client
    
    # Startup
    logger.info("Starting Market Data Service...")
    
    # Initialize Kafka producer
    kafka_producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
        retries=5,
        retry_backoff_ms=1000
    )
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://redis:6379")
    
    # Start market data streaming
    asyncio.create_task(stream_market_data())
    
    logger.info("Market Data Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Market Data Service...")
    if kafka_producer:
        kafka_producer.close()
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="FinSim Market Data Service",
    description="Real-time and historical market data streaming service",
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

# Market Data Connectors
class YahooFinanceConnector:
    """Yahoo Finance data connector"""
    
    @staticmethod
    def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_real_time_quote(symbol: str) -> Optional[QuoteData]:
        """Fetch real-time quote from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return QuoteData(
                symbol=symbol,
                price=info.get('currentPrice', 0.0),
                bid=info.get('bid', 0.0),
                ask=info.get('ask', 0.0),
                volume=info.get('volume', 0),
                timestamp=datetime.now(),
                exchange=info.get('exchange', 'NYSE')
            )
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None

class BloombergConnector:
    """Bloomberg API connector (simulation for demo)"""
    
    @staticmethod
    def get_real_time_quote(symbol: str) -> Optional[QuoteData]:
        """Simulate Bloomberg real-time data"""
        # In production, this would use the actual Bloomberg API
        base_price = 100.0
        volatility = 0.02
        price = base_price * (1 + np.random.normal(0, volatility))
        
        return QuoteData(
            symbol=symbol,
            price=round(price, 2),
            bid=round(price * 0.999, 2),
            ask=round(price * 1.001, 2),
            volume=np.random.randint(1000, 10000),
            timestamp=datetime.now(),
            exchange="NYSE"
        )

class PolygonConnector:
    """Polygon.io connector (simulation for demo)"""
    
    @staticmethod
    def get_real_time_quote(symbol: str) -> Optional[QuoteData]:
        """Simulate Polygon real-time data"""
        # In production, this would use the actual Polygon.io API
        base_price = 150.0
        volatility = 0.015
        price = base_price * (1 + np.random.normal(0, volatility))
        
        return QuoteData(
            symbol=symbol,
            price=round(price, 2),
            bid=round(price * 0.9995, 2),
            ask=round(price * 1.0005, 2),
            volume=np.random.randint(5000, 50000),
            timestamp=datetime.now(),
            exchange="NASDAQ"
        )

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/v1/quotes/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    try:
        # Try to get from cache first
        cached_quote = await redis_client.get(f"quote:{symbol}")
        if cached_quote:
            return json.loads(cached_quote)
        
        # Fetch from data provider
        quote = YahooFinanceConnector.get_real_time_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, message=f"Quote not found for symbol {symbol}")
        
        # Cache the quote
        await redis_client.setex(
            f"quote:{symbol}", 
            30,  # Cache for 30 seconds
            quote.model_dump_json()
        )
        
        return quote.model_dump()
    
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/historical")
async def get_historical_data(request: MarketDataRequest):
    """Get historical market data for symbols"""
    try:
        results = {}
        
        for symbol in request.symbols:
            data = YahooFinanceConnector.get_historical_data(
                symbol=symbol,
                period="1y",  # You can make this configurable based on request
                interval=request.interval
            )
            
            if not data.empty:
                results[symbol] = {
                    "data": data.reset_index().to_dict(orient="records"),
                    "symbol": symbol,
                    "interval": request.interval
                }
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/fundamentals/{symbol}")
async def get_fundamentals(symbol: str):
    """Get fundamental data for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = FundamentalData(
            symbol=symbol,
            pe_ratio=info.get('trailingPE'),
            pb_ratio=info.get('priceToBook'),
            debt_to_equity=info.get('debtToEquity'),
            roe=info.get('returnOnEquity'),
            revenue=info.get('totalRevenue'),
            net_income=info.get('netIncomeToCommon'),
            timestamp=datetime.now()
        )
        
        return fundamentals.model_dump()
    
    except Exception as e:
        logger.error(f"Error getting fundamentals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/market-data")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

async def stream_market_data():
    """Background task to stream market data to Kafka and WebSocket clients"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
    
    while True:
        try:
            for symbol in symbols:
                # Generate real-time quote
                quote = BloombergConnector.get_real_time_quote(symbol)
                if quote:
                    quote_data = quote.model_dump()
                    
                    # Send to Kafka
                    if kafka_producer:
                        kafka_producer.send('prices', value=quote_data)
                    
                    # Cache in Redis
                    if redis_client:
                        await redis_client.setex(
                            f"quote:{symbol}",
                            60,  # Cache for 1 minute
                            json.dumps(quote_data, default=str)
                        )
                    
                    # Send to WebSocket clients
                    for websocket in websocket_connections[:]:  # Create a copy to avoid modification during iteration
                        try:
                            await websocket.send_json(quote_data)
                        except:
                            websocket_connections.remove(websocket)
            
            # Wait before next update
            await asyncio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Error in market data streaming: {e}")
            await asyncio.sleep(5)  # Wait longer on error

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )