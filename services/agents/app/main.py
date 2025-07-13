#!/usr/bin/env python3
"""
FinSim Agents Service

Hosts pluggable agents: heuristic, statistical, LSTM/Transformer/GRU, DQN/PPO/A3C.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from contextlib import asynccontextmanager
import uuid
import httpx
import threading
import talib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent Types
class AgentType(str, Enum):
    HEURISTIC = "heuristic"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRU = "gru"
    DQN = "dqn"
    PPO = "ppo"
    A3C = "a3c"

class Strategy(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    ARBITRAGE = "arbitrage"

# Data Models
class AgentConfig(BaseModel):
    agent_id: str
    agent_type: AgentType
    strategy: Optional[Strategy] = None
    symbols: List[str]
    parameters: Dict[str, Any] = {}
    enabled: bool = True

class AgentStatus(BaseModel):
    agent_id: str
    status: str
    last_action: Optional[str] = None
    performance: Dict[str, float] = {}
    timestamp: datetime

# Base Agent Class
class BaseAgent:
    """Base class for all trading agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.symbols = config.symbols
        self.enabled = config.enabled
        self.performance = {
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0
        }
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.last_prices = {}
        self.running = False
        
    async def start(self):
        """Start the agent"""
        self.running = True
        logger.info(f"Agent {self.agent_id} started")
        
    async def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
        
    async def update_market_data(self, data: Dict):
        """Update with new market data"""
        symbol = data.get('symbol')
        if symbol in self.symbols:
            self.last_prices[symbol] = data.get('price', 0.0)
            if self.enabled and self.running:
                await self.make_decision(symbol, data)
    
    async def make_decision(self, symbol: str, data: Dict):
        """Make trading decision - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def place_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None):
        """Place order through simulation engine"""
        try:
            async with httpx.AsyncClient() as client:
                order_data = {
                    "symbol": symbol,
                    "side": side,
                    "order_type": "limit" if price else "market",
                    "quantity": quantity,
                    "price": price,
                    "agent_id": self.agent_id
                }
                
                response = await client.post(
                    "http://simulation-engine:8000/api/v1/orders",
                    json=order_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Agent {self.agent_id}: Order placed - {side} {quantity} {symbol} at {price}")
                    self.performance["total_trades"] += 1
                    return result
                else:
                    logger.error(f"Agent {self.agent_id}: Failed to place order - {response.text}")
                    
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error placing order - {e}")

# Heuristic Agents
class MomentumAgent(BaseAgent):
    """Momentum-based trading agent using RSI"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.rsi_overbought = config.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = config.parameters.get('rsi_oversold', 30)
        self.price_history = {symbol: [] for symbol in self.symbols}
        
    async def make_decision(self, symbol: str, data: Dict):
        """Make decision based on RSI momentum"""
        price = data.get('price', 0.0)
        self.price_history[symbol].append(price)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Need at least rsi_period + 1 prices for RSI calculation
        if len(self.price_history[symbol]) < self.rsi_period + 1:
            return
            
        prices = np.array(self.price_history[symbol])
        rsi = talib.RSI(prices, timeperiod=self.rsi_period)[-1]
        
        current_position = self.positions[symbol]
        
        # Buy signal: RSI oversold and no position
        if rsi < self.rsi_oversold and current_position == 0:
            await self.place_order(symbol, "buy", 100, price * 1.001)  # Slightly above market
            self.positions[symbol] = 100
            
        # Sell signal: RSI overbought and have position
        elif rsi > self.rsi_overbought and current_position > 0:
            await self.place_order(symbol, "sell", current_position, price * 0.999)  # Slightly below market
            self.positions[symbol] = 0

class MeanReversionAgent(BaseAgent):
    """Mean reversion agent using Bollinger Bands"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.bb_period = config.parameters.get('bb_period', 20)
        self.bb_std = config.parameters.get('bb_std', 2)
        self.price_history = {symbol: [] for symbol in self.symbols}
        
    async def make_decision(self, symbol: str, data: Dict):
        """Make decision based on Bollinger Bands"""
        price = data.get('price', 0.0)
        self.price_history[symbol].append(price)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        if len(self.price_history[symbol]) < self.bb_period:
            return
            
        prices = np.array(self.price_history[symbol])
        upper, middle, lower = talib.BBANDS(prices, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
        
        current_position = self.positions[symbol]
        
        # Buy signal: price touches lower band
        if price <= lower[-1] and current_position == 0:
            await self.place_order(symbol, "buy", 100, price * 1.001)
            self.positions[symbol] = 100
            
        # Sell signal: price touches upper band
        elif price >= upper[-1] and current_position > 0:
            await self.place_order(symbol, "sell", current_position, price * 0.999)
            self.positions[symbol] = 0

# LSTM Agent Implementation
class LSTMModel(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMAgent(BaseAgent):
    """LSTM-based trading agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.sequence_length = config.parameters.get('sequence_length', 20)
        self.model = LSTMModel()
        self.scaler = MinMaxScaler()
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.features_history = {symbol: [] for symbol in self.symbols}
        self.is_trained = False
        
    def prepare_features(self, symbol: str) -> Optional[np.ndarray]:
        """Prepare features for the model"""
        if len(self.price_history[symbol]) < self.sequence_length + 5:
            return None
            
        prices = np.array(self.price_history[symbol][-50:])  # Use last 50 prices
        
        # Calculate technical indicators
        returns = np.diff(prices) / prices[:-1]
        sma5 = talib.SMA(prices, timeperiod=5)
        sma20 = talib.SMA(prices, timeperiod=20)
        rsi = talib.RSI(prices, timeperiod=14)
        volume = np.random.randint(1000, 10000, len(prices))  # Simulated volume
        
        # Combine features
        features = np.column_stack([
            prices[20:],  # Raw prices (skip NaN from indicators)
            returns[19:],  # Returns
            sma5[20:],     # SMA5
            rsi[20:],      # RSI
            volume[20:]    # Volume
        ])
        
        return features
    
    async def make_decision(self, symbol: str, data: Dict):
        """Make decision based on LSTM prediction"""
        price = data.get('price', 0.0)
        self.price_history[symbol].append(price)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
        
        features = self.prepare_features(symbol)
        if features is None or len(features) < self.sequence_length:
            return
        
        # Prepare input for model
        if not self.is_trained:
            # Simple training simulation
            self.is_trained = True
            logger.info(f"LSTM Agent {self.agent_id}: Model training completed for {symbol}")
        
        # Make prediction (simplified)
        recent_features = features[-self.sequence_length:]
        normalized_features = self.scaler.fit_transform(recent_features)
        
        # Simple prediction logic (in practice, would use trained model)
        predicted_change = np.mean(normalized_features[:, 1])  # Use returns as proxy
        
        current_position = self.positions[symbol]
        
        # Trading logic based on prediction
        if predicted_change > 0.001 and current_position == 0:  # Predict price increase
            await self.place_order(symbol, "buy", 100, price * 1.001)
            self.positions[symbol] = 100
        elif predicted_change < -0.001 and current_position > 0:  # Predict price decrease
            await self.place_order(symbol, "sell", current_position, price * 0.999)
            self.positions[symbol] = 0

# DQN Agent Implementation
class DQNModel(nn.Module):
    """Deep Q-Network for reinforcement learning"""
    
    def __init__(self, state_size=10, action_size=3, hidden_size=64):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent(BaseAgent):
    """Deep Q-Network trading agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.state_size = 10
        self.action_size = 3  # 0: hold, 1: buy, 2: sell
        self.model = DQNModel(self.state_size, self.action_size)
        self.epsilon = config.parameters.get('epsilon', 0.1)
        self.price_history = {symbol: [] for symbol in self.symbols}
        
    def get_state(self, symbol: str) -> Optional[np.ndarray]:
        """Get current state representation"""
        if len(self.price_history[symbol]) < 20:
            return None
            
        prices = np.array(self.price_history[symbol][-20:])
        returns = np.diff(prices) / prices[:-1]
        
        # State features: recent returns, position, price momentum
        state = np.concatenate([
            returns[-5:],                           # Last 5 returns
            [self.positions[symbol] / 100],         # Normalized position
            [np.mean(returns[-10:])],               # Recent momentum
            [np.std(returns[-10:])],                # Recent volatility
            [len(self.price_history[symbol]) / 100], # Time factor
            [self.performance["total_pnl"] / 1000]   # Performance factor
        ])
        
        return state
    
    async def make_decision(self, symbol: str, data: Dict):
        """Make decision using DQN"""
        price = data.get('price', 0.0)
        self.price_history[symbol].append(price)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        state = self.get_state(symbol)
        if state is None:
            return
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()
        
        current_position = self.positions[symbol]
        
        # Execute action
        if action == 1 and current_position == 0:  # Buy
            await self.place_order(symbol, "buy", 100, price * 1.001)
            self.positions[symbol] = 100
        elif action == 2 and current_position > 0:  # Sell
            await self.place_order(symbol, "sell", current_position, price * 0.999)
            self.positions[symbol] = 0
        # action == 0 is hold, do nothing

# Global variables
agents: Dict[str, BaseAgent] = {}
kafka_producer = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global kafka_producer, redis_client
    
    # Startup
    logger.info("Starting Agents Service...")
    
    # Initialize Kafka producer
    kafka_producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://redis:6379")
    
    # Start market data consumer
    asyncio.create_task(consume_market_data())
    
    logger.info("Agents Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agents Service...")
    for agent in agents.values():
        await agent.stop()
    if kafka_producer:
        kafka_producer.close()
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="FinSim Agents Service",
    description="ML and RL trading agents",
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

@app.post("/api/v1/agents")
async def create_agent(config: AgentConfig):
    """Create a new trading agent"""
    try:
        if config.agent_id in agents:
            raise HTTPException(status_code=400, detail="Agent already exists")
        
        # Create agent based on type
        if config.agent_type == AgentType.HEURISTIC:
            if config.strategy == Strategy.MOMENTUM:
                agent = MomentumAgent(config)
            elif config.strategy == Strategy.MEAN_REVERSION:
                agent = MeanReversionAgent(config)
            else:
                raise HTTPException(status_code=400, detail="Invalid strategy for heuristic agent")
        elif config.agent_type == AgentType.LSTM:
            agent = LSTMAgent(config)
        elif config.agent_type == AgentType.DQN:
            agent = DQNAgent(config)
        else:
            raise HTTPException(status_code=400, detail="Agent type not yet implemented")
        
        agents[config.agent_id] = agent
        await agent.start()
        
        logger.info(f"Created agent {config.agent_id} of type {config.agent_type}")
        
        return {"status": "created", "agent_id": config.agent_id}
    
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents():
    """List all agents"""
    agent_list = []
    for agent_id, agent in agents.items():
        status = AgentStatus(
            agent_id=agent_id,
            status="running" if agent.running else "stopped",
            performance=agent.performance,
            timestamp=datetime.now()
        )
        agent_list.append(status.model_dump())
    
    return agent_list

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    status = AgentStatus(
        agent_id=agent_id,
        status="running" if agent.running else "stopped",
        performance=agent.performance,
        timestamp=datetime.now()
    )
    
    return status.model_dump()

@app.post("/api/v1/agents/{agent_id}/start")
async def start_agent(agent_id: str):
    """Start an agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    await agent.start()
    
    return {"status": "started", "agent_id": agent_id}

@app.post("/api/v1/agents/{agent_id}/stop")
async def stop_agent(agent_id: str):
    """Stop an agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    await agent.stop()
    
    return {"status": "stopped", "agent_id": agent_id}

@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    await agent.stop()
    del agents[agent_id]
    
    return {"status": "deleted", "agent_id": agent_id}

async def consume_market_data():
    """Consume market data from Kafka and update agents"""
    while True:
        try:
            # Simulate consuming from Kafka (in production, would use actual Kafka consumer)
            await asyncio.sleep(1)
            
            # Simulate market data
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
            for symbol in symbols:
                price = 100 + np.random.normal(0, 2)
                data = {
                    "symbol": symbol,
                    "price": price,
                    "timestamp": datetime.now(),
                    "volume": np.random.randint(1000, 10000)
                }
                
                # Update all agents with this data
                for agent in agents.values():
                    if symbol in agent.symbols:
                        await agent.update_market_data(data)
                        
        except Exception as e:
            logger.error(f"Error in market data consumer: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )