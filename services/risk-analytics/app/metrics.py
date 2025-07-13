#!/usr/bin/env python3
"""
FinSim Risk Analytics Service

Computes VaR (Historical, Parametric, Monte Carlo) & Expected Shortfall per Basel III.
Implements volatility, Sharpe, Sortino, Calmar ratios and Max Drawdown calculations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import sqlalchemy
from sqlalchemy import create_engine, text
from contextlib import asynccontextmanager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Risk Models
class RiskMetricType(str, Enum):
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    EXPECTED_SHORTFALL = "expected_shortfall"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"

class ConfidenceLevel(float, Enum):
    NINETY_FIVE = 0.95
    NINETY_NINE = 0.99
    NINETY_NINE_NINE = 0.999

# Data Models
class PortfolioPosition(BaseModel):
    symbol: str
    quantity: int
    current_price: float
    market_value: float
    weight: float

class PortfolioData(BaseModel):
    portfolio_id: str
    positions: List[PortfolioPosition]
    total_value: float
    timestamp: datetime

class RiskMetricRequest(BaseModel):
    portfolio_id: str
    metric_type: RiskMetricType
    confidence_level: Optional[ConfidenceLevel] = ConfidenceLevel.NINETY_FIVE
    time_horizon_days: Optional[int] = 1
    lookback_days: Optional[int] = 252
    simulations: Optional[int] = 10000

class RiskMetricResult(BaseModel):
    portfolio_id: str
    metric_type: RiskMetricType
    value: float
    confidence_level: Optional[float] = None
    currency: str = "USD"
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class VaRResult(BaseModel):
    portfolio_id: str
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    method: str
    currency: str = "USD"
    timestamp: datetime

# Risk Calculator Class
class RiskCalculator:
    """Basel III compliant risk calculations"""
    
    @staticmethod
    def historical_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Historical Value at Risk calculation
        Basel III FAQ #12 compliant
        """
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate percentile
        percentile = (1 - confidence_level) * 100
        var = np.percentile(sorted_returns, percentile)
        
        return -var  # Return positive value for loss
    
    @staticmethod
    def parametric_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Parametric (Normal distribution) Value at Risk
        Assumes returns are normally distributed
        """
        if len(returns) == 0:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var = -(mean + z_score * std)
        
        return var
    
    @staticmethod
    def monte_carlo_var(returns: np.ndarray, confidence_level: float = 0.95, 
                       simulations: int = 10000) -> float:
        """
        Monte Carlo Value at Risk simulation
        """
        if len(returns) == 0:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mean, std, simulations)
        
        # Calculate VaR from simulated returns
        var = RiskCalculator.historical_var(simulated_returns, confidence_level)
        
        return var
    
    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Expected Shortfall (Conditional VaR) calculation
        Basel III compliant - average of losses beyond VaR
        """
        if len(returns) == 0:
            return 0.0
        
        var = RiskCalculator.historical_var(returns, confidence_level)
        
        # Find returns worse than VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        # Expected Shortfall is the average of tail losses
        es = -np.mean(tail_returns)
        
        return es
    
    @staticmethod
    def volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        """
        if len(returns) == 0:
            return 0.0
        
        vol = np.std(returns, ddof=1)
        
        if annualize:
            # Annualize assuming 252 trading days
            vol *= np.sqrt(252)
        
        return vol
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe ratio calculation
        (Mean return - Risk-free rate) / Volatility
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns) * 252  # Annualized
        volatility = RiskCalculator.volatility(returns, annualize=True)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / volatility
        
        return sharpe
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sortino ratio calculation
        Uses downside deviation instead of total volatility
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns) * 252  # Annualized
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = (mean_return - risk_free_rate) / downside_vol
        
        return sortino
    
    @staticmethod
    def max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
        """
        Maximum Drawdown calculation (vectorized)
        Returns: (max_dd, start_idx, end_idx)
        """
        if len(prices) == 0:
            return 0.0, 0, 0
        
        # Calculate cumulative maximum (peak)
        peak = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find the peak before max drawdown
        peak_idx = np.argmax(peak[:max_dd_idx + 1])
        
        return -max_dd, peak_idx, max_dd_idx
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, prices: np.ndarray) -> float:
        """
        Calmar ratio calculation
        Annualized return / Maximum Drawdown
        """
        if len(returns) == 0 or len(prices) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd, _, _ = RiskCalculator.max_drawdown(prices)
        
        if max_dd == 0:
            return float('inf')
        
        calmar = annual_return / max_dd
        
        return calmar

# Portfolio Risk Analytics
class PortfolioRiskAnalytics:
    """Portfolio-level risk analytics"""
    
    def __init__(self, db_engine):
        self.db_engine = db_engine
    
    async def get_portfolio_returns(self, portfolio_id: str, days: int = 252) -> pd.DataFrame:
        """Get historical returns for portfolio"""
        try:
            # In production, this would query actual portfolio returns
            # For demo, we'll simulate portfolio returns
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Simulate correlated returns for portfolio
            returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
            
            df = pd.DataFrame({
                'date': dates,
                'portfolio_return': returns,
                'portfolio_value': 1000000 * (1 + returns).cumprod()
            })
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {e}")
            return pd.DataFrame()
    
    async def calculate_portfolio_var(self, portfolio_id: str, 
                                    confidence_level: float = 0.95,
                                    method: str = "historical") -> VaRResult:
        """Calculate portfolio VaR using specified method"""
        try:
            # Get portfolio returns
            returns_df = await self.get_portfolio_returns(portfolio_id)
            
            if returns_df.empty:
                raise HTTPException(status_code=404, detail="No data found for portfolio")
            
            returns = returns_df['portfolio_return'].values
            
            # Calculate VaR using different methods
            if method == "historical":
                var_95 = RiskCalculator.historical_var(returns, 0.95)
                var_99 = RiskCalculator.historical_var(returns, 0.99)
            elif method == "parametric":
                var_95 = RiskCalculator.parametric_var(returns, 0.95)
                var_99 = RiskCalculator.parametric_var(returns, 0.99)
            elif method == "monte_carlo":
                var_95 = RiskCalculator.monte_carlo_var(returns, 0.95)
                var_99 = RiskCalculator.monte_carlo_var(returns, 0.99)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            # Calculate Expected Shortfall
            es_95 = RiskCalculator.expected_shortfall(returns, 0.95)
            es_99 = RiskCalculator.expected_shortfall(returns, 0.99)
            
            # Convert to dollar amounts (assuming $1M portfolio)
            portfolio_value = 1000000
            
            result = VaRResult(
                portfolio_id=portfolio_id,
                var_95=var_95 * portfolio_value,
                var_99=var_99 * portfolio_value,
                expected_shortfall_95=es_95 * portfolio_value,
                expected_shortfall_99=es_99 * portfolio_value,
                method=method,
                timestamp=datetime.now()
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def calculate_performance_metrics(self, portfolio_id: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            returns_df = await self.get_portfolio_returns(portfolio_id)
            
            if returns_df.empty:
                return {}
            
            returns = returns_df['portfolio_return'].values
            prices = returns_df['portfolio_value'].values
            
            metrics = {
                "volatility": RiskCalculator.volatility(returns),
                "sharpe_ratio": RiskCalculator.sharpe_ratio(returns),
                "sortino_ratio": RiskCalculator.sortino_ratio(returns),
                "calmar_ratio": RiskCalculator.calmar_ratio(returns, prices),
                "max_drawdown": RiskCalculator.max_drawdown(prices)[0],
                "var_95": RiskCalculator.historical_var(returns, 0.95),
                "var_99": RiskCalculator.historical_var(returns, 0.99),
                "expected_shortfall_95": RiskCalculator.expected_shortfall(returns, 0.95),
                "expected_shortfall_99": RiskCalculator.expected_shortfall(returns, 0.99)
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

# Global variables
portfolio_analytics = None
kafka_producer = None
redis_client = None
db_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global portfolio_analytics, kafka_producer, redis_client, db_engine
    
    # Startup
    logger.info("Starting Risk Analytics Service...")
    
    # Initialize database connection
    db_engine = create_engine("postgresql://finsim:finsim123@postgres:5432/finsim")
    portfolio_analytics = PortfolioRiskAnalytics(db_engine)
    
    # Initialize Kafka producer
    kafka_producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://redis:6379")
    
    logger.info("Risk Analytics Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Risk Analytics Service...")
    if kafka_producer:
        kafka_producer.close()
    if redis_client:
        await redis_client.close()
    if db_engine:
        db_engine.dispose()

app = FastAPI(
    title="FinSim Risk Analytics Service",
    description="Comprehensive risk analytics and VaR calculations",
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

@app.post("/api/v1/risk/var")
async def calculate_var(request: RiskMetricRequest):
    """Calculate Value at Risk for a portfolio"""
    try:
        method_map = {
            RiskMetricType.VAR_HISTORICAL: "historical",
            RiskMetricType.VAR_PARAMETRIC: "parametric",
            RiskMetricType.VAR_MONTE_CARLO: "monte_carlo"
        }
        
        method = method_map.get(request.metric_type, "historical")
        
        result = await portfolio_analytics.calculate_portfolio_var(
            portfolio_id=request.portfolio_id,
            confidence_level=request.confidence_level,
            method=method
        )
        
        return result.model_dump()
    
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/metrics/{portfolio_id}")
async def get_risk_metrics(portfolio_id: str):
    """Get comprehensive risk metrics for a portfolio"""
    try:
        metrics = await portfolio_analytics.calculate_performance_metrics(portfolio_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Portfolio not found or no data available")
        
        return {
            "portfolio_id": portfolio_id,
            "metrics": metrics,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/stress-test/{portfolio_id}")
async def stress_test(portfolio_id: str, 
                     scenario: str = Query("market_crash", description="Stress test scenario")):
    """Perform stress testing on portfolio"""
    try:
        returns_df = await portfolio_analytics.get_portfolio_returns(portfolio_id)
        
        if returns_df.empty:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        base_returns = returns_df['portfolio_return'].values
        current_value = 1000000  # Assuming $1M portfolio
        
        # Define stress scenarios
        scenarios = {
            "market_crash": {"shock": -0.20, "volatility_multiplier": 2.0},
            "high_volatility": {"shock": 0.0, "volatility_multiplier": 3.0},
            "recession": {"shock": -0.10, "volatility_multiplier": 1.5},
            "inflation_spike": {"shock": -0.05, "volatility_multiplier": 1.2}
        }
        
        if scenario not in scenarios:
            raise HTTPException(status_code=400, detail="Invalid scenario")
        
        scenario_params = scenarios[scenario]
        
        # Apply stress scenario
        shocked_returns = base_returns * scenario_params["volatility_multiplier"] + scenario_params["shock"]
        
        # Calculate stressed metrics
        stressed_var_95 = RiskCalculator.historical_var(shocked_returns, 0.95) * current_value
        stressed_var_99 = RiskCalculator.historical_var(shocked_returns, 0.99) * current_value
        stressed_es_95 = RiskCalculator.expected_shortfall(shocked_returns, 0.95) * current_value
        
        # Calculate portfolio value change
        portfolio_change = np.sum(shocked_returns) * current_value
        
        result = {
            "portfolio_id": portfolio_id,
            "scenario": scenario,
            "portfolio_value_change": portfolio_change,
            "stressed_var_95": stressed_var_95,
            "stressed_var_99": stressed_var_99,
            "stressed_expected_shortfall_95": stressed_es_95,
            "scenario_parameters": scenario_params,
            "timestamp": datetime.now()
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/correlation/{portfolio_id}")
async def get_correlation_analysis(portfolio_id: str):
    """Get correlation analysis for portfolio holdings"""
    try:
        # Simulate correlation matrix for portfolio holdings
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        # Generate realistic correlation matrix
        np.random.seed(42)
        correlation_matrix = np.random.uniform(0.3, 0.8, (len(symbols), len(symbols)))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Convert to dict format
        correlation_data = {}
        for i, symbol1 in enumerate(symbols):
            correlation_data[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                correlation_data[symbol1][symbol2] = round(correlation_matrix[i, j], 3)
        
        return {
            "portfolio_id": portfolio_id,
            "correlation_matrix": correlation_data,
            "symbols": symbols,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/risk/backtesting")
async def backtest_var_model(portfolio_id: str, 
                           confidence_level: float = 0.95,
                           method: str = "historical"):
    """Backtest VaR model accuracy"""
    try:
        returns_df = await portfolio_analytics.get_portfolio_returns(portfolio_id, days=500)
        
        if returns_df.empty:
            raise HTTPException(status_code=404, detail="Insufficient data for backtesting")
        
        returns = returns_df['portfolio_return'].values
        
        # Perform rolling VaR backtesting
        window_size = 252
        violations = 0
        total_observations = 0
        
        for i in range(window_size, len(returns)):
            # Calculate VaR using historical window
            historical_returns = returns[i-window_size:i]
            
            if method == "historical":
                var = RiskCalculator.historical_var(historical_returns, confidence_level)
            elif method == "parametric":
                var = RiskCalculator.parametric_var(historical_returns, confidence_level)
            else:
                var = RiskCalculator.monte_carlo_var(historical_returns, confidence_level)
            
            # Check if actual return violates VaR
            actual_return = returns[i]
            if actual_return < -var:
                violations += 1
            
            total_observations += 1
        
        # Calculate statistics
        violation_rate = violations / total_observations if total_observations > 0 else 0
        expected_violations = 1 - confidence_level
        
        # Kupiec Test (simple version)
        likelihood_ratio = -2 * np.log(
            (expected_violations ** violations) * ((1 - expected_violations) ** (total_observations - violations))
        ) if violations > 0 else 0
        
        result = {
            "portfolio_id": portfolio_id,
            "method": method,
            "confidence_level": confidence_level,
            "total_observations": total_observations,
            "violations": violations,
            "violation_rate": violation_rate,
            "expected_violation_rate": expected_violations,
            "kupiec_test_statistic": likelihood_ratio,
            "model_accuracy": "Good" if abs(violation_rate - expected_violations) < 0.02 else "Poor",
            "timestamp": datetime.now()
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in VaR backtesting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "metrics:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )