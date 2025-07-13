#!/usr/bin/env python3
"""
FinSim Portfolio Service

Mean-variance and Black-Litterman portfolio optimization.
Implements efficient frontier calculation and portfolio rebalancing.
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
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy import linalg
import sqlalchemy
from sqlalchemy import create_engine, text
from contextlib import asynccontextmanager
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portfolio Models
class OptimizationMethod(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"

class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

# Data Models
class AssetData(BaseModel):
    symbol: str
    expected_return: float
    volatility: float
    current_price: float
    market_cap: Optional[float] = None

class PortfolioConstraints(BaseModel):
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_concentration: float = 0.3
    min_positions: int = 3
    max_positions: int = 20
    target_risk: Optional[float] = None
    target_return: Optional[float] = None

class OptimizationRequest(BaseModel):
    portfolio_id: str
    assets: List[AssetData]
    method: OptimizationMethod
    constraints: PortfolioConstraints
    risk_aversion: Optional[float] = 1.0
    confidence_level: Optional[float] = 0.95

class PortfolioAllocation(BaseModel):
    symbol: str
    weight: float
    shares: int
    market_value: float

class OptimizationResult(BaseModel):
    portfolio_id: str
    method: OptimizationMethod
    allocations: List[PortfolioAllocation]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    total_value: float
    timestamp: datetime

class EfficientFrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]

class EfficientFrontier(BaseModel):
    portfolio_id: str
    points: List[EfficientFrontierPoint]
    optimal_portfolio: EfficientFrontierPoint
    timestamp: datetime

# Portfolio Optimizer Class
class PortfolioOptimizer:
    """Portfolio optimization engine"""
    
    def __init__(self, db_engine):
        self.db_engine = db_engine
    
    @staticmethod
    def estimate_returns_covariance(symbols: List[str], lookback_days: int = 252) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected returns and covariance matrix"""
        # In production, this would fetch actual historical data
        # For demo, we'll simulate realistic returns and correlations
        
        n_assets = len(symbols)
        
        # Simulate expected returns (annualized)
        expected_returns = np.random.uniform(0.05, 0.15, n_assets)
        
        # Simulate covariance matrix
        # Start with random correlations
        correlations = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
        np.fill_diagonal(correlations, 1.0)
        
        # Make symmetric
        correlations = (correlations + correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Ensure positive definite
        eigenvals, eigenvecs = linalg.eigh(correlations)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Convert to covariance using random volatilities
        volatilities = np.random.uniform(0.15, 0.35, n_assets)
        covariance = np.outer(volatilities, volatilities) * correlations
        
        return expected_returns, covariance
    
    def mean_variance_optimization(self, 
                                 expected_returns: np.ndarray,
                                 covariance: np.ndarray,
                                 risk_aversion: float = 1.0,
                                 constraints: PortfolioConstraints = None) -> np.ndarray:
        """
        Mean-variance optimization using quadratic programming
        Maximizes: w'μ - (λ/2)w'Σw
        """
        n_assets = len(expected_returns)
        
        # Decision variables: portfolio weights
        w = cp.Variable(n_assets)
        
        # Objective: maximize utility (return - risk penalty)
        portfolio_return = expected_returns.T @ w
        portfolio_risk = cp.quad_form(w, covariance)
        objective = cp.Maximize(portfolio_return - (risk_aversion / 2) * portfolio_risk)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only positions
        ]
        
        if constraints:
            if constraints.min_weight > 0:
                constraints_list.append(w >= constraints.min_weight)
            if constraints.max_weight < 1:
                constraints_list.append(w <= constraints.max_weight)
            if constraints.max_concentration < 1:
                constraints_list.append(w <= constraints.max_concentration)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed: {problem.status}")
        
        return w.value
    
    def black_litterman_optimization(self,
                                   expected_returns: np.ndarray,
                                   covariance: np.ndarray,
                                   market_caps: np.ndarray,
                                   views: Optional[np.ndarray] = None,
                                   view_uncertainty: Optional[np.ndarray] = None,
                                   risk_aversion: float = 3.0) -> np.ndarray:
        """
        Black-Litterman portfolio optimization
        Combines market equilibrium with investor views
        """
        n_assets = len(expected_returns)
        
        # Market portfolio weights (based on market cap)
        if market_caps is not None:
            w_market = market_caps / np.sum(market_caps)
        else:
            w_market = np.ones(n_assets) / n_assets
        
        # Implied equilibrium returns
        pi = risk_aversion * covariance @ w_market
        
        # If no views provided, use equilibrium
        if views is None:
            mu_bl = pi
            sigma_bl = covariance
        else:
            # Black-Litterman formula
            tau = 0.05  # Scaling factor for uncertainty of prior
            
            if view_uncertainty is None:
                # Default view uncertainty
                view_uncertainty = np.diag(np.diag(covariance)) * 0.1
            
            # BL expected returns
            sigma_bl_inv = linalg.inv(tau * covariance) + views.T @ linalg.inv(view_uncertainty) @ views
            sigma_bl = linalg.inv(sigma_bl_inv)
            
            mu_bl = sigma_bl @ (linalg.inv(tau * covariance) @ pi + views.T @ linalg.inv(view_uncertainty) @ views @ expected_returns)
        
        # Optimize with BL inputs
        return self.mean_variance_optimization(mu_bl, sigma_bl, risk_aversion)
    
    def risk_parity_optimization(self, covariance: np.ndarray) -> np.ndarray:
        """
        Risk parity optimization - equal risk contribution
        """
        n_assets = len(covariance)
        
        def risk_parity_objective(weights):
            """Objective function for risk parity"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ covariance @ weights)
            
            # Risk contributions
            risk_contrib = (weights * (covariance @ weights)) / portfolio_vol
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            # Minimize squared deviations from equal risk
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Reasonable bounds
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Risk parity optimization did not converge, using equal weights")
            return np.ones(n_assets) / n_assets
        
        return result.x
    
    def minimum_variance_optimization(self, covariance: np.ndarray) -> np.ndarray:
        """Minimum variance portfolio optimization"""
        n_assets = len(covariance)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, covariance))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Minimum variance optimization failed: {problem.status}")
        
        return w.value
    
    def maximum_sharpe_optimization(self, 
                                  expected_returns: np.ndarray,
                                  covariance: np.ndarray,
                                  risk_free_rate: float = 0.02) -> np.ndarray:
        """Maximum Sharpe ratio portfolio optimization"""
        n_assets = len(expected_returns)
        
        # Transform to auxiliary variables to make linear
        # Maximize (w'μ - rf) / sqrt(w'Σw)
        # Equivalent to maximizing (w'μ - rf*sum(w)) subject to w'Σw <= 1
        
        w = cp.Variable(n_assets)
        kappa = cp.Variable()  # Auxiliary variable
        
        # Objective: maximize excess return
        excess_returns = expected_returns - risk_free_rate
        objective = cp.Maximize(excess_returns.T @ w)
        
        # Constraints
        constraints = [
            cp.quad_form(w, covariance) <= 1,  # Risk constraint
            cp.sum(w) == kappa,  # Normalization
            w >= 0  # Long-only
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Maximum Sharpe optimization failed: {problem.status}")
        
        # Normalize weights
        weights = w.value / kappa.value
        
        return weights
    
    def calculate_efficient_frontier(self,
                                   expected_returns: np.ndarray,
                                   covariance: np.ndarray,
                                   symbols: List[str],
                                   n_points: int = 50) -> List[EfficientFrontierPoint]:
        """Calculate efficient frontier points"""
        
        # Range of target returns
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_points = []
        
        for target_return in target_returns:
            try:
                # Optimize for target return
                n_assets = len(expected_returns)
                w = cp.Variable(n_assets)
                
                # Minimize variance for target return
                objective = cp.Minimize(cp.quad_form(w, covariance))
                constraints = [
                    cp.sum(w) == 1,
                    expected_returns.T @ w == target_return,
                    w >= 0
                ]
                
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                if problem.status == cp.OPTIMAL:
                    weights = w.value
                    portfolio_return = target_return
                    portfolio_vol = np.sqrt(weights.T @ covariance @ weights)
                    sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                    
                    point = EfficientFrontierPoint(
                        expected_return=portfolio_return,
                        volatility=portfolio_vol,
                        sharpe_ratio=sharpe_ratio,
                        weights=dict(zip(symbols, weights))
                    )
                    frontier_points.append(point)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate frontier point for return {target_return}: {e}")
                continue
        
        return frontier_points
    
    async def optimize_portfolio(self, request: OptimizationRequest) -> OptimizationResult:
        """Main portfolio optimization function"""
        try:
            symbols = [asset.symbol for asset in request.assets]
            
            # Get returns and covariance
            expected_returns, covariance = self.estimate_returns_covariance(symbols)
            
            # Perform optimization based on method
            if request.method == OptimizationMethod.MEAN_VARIANCE:
                weights = self.mean_variance_optimization(
                    expected_returns, covariance, 
                    request.risk_aversion, request.constraints
                )
            elif request.method == OptimizationMethod.BLACK_LITTERMAN:
                market_caps = np.array([asset.market_cap or 1e9 for asset in request.assets])
                weights = self.black_litterman_optimization(
                    expected_returns, covariance, market_caps, 
                    risk_aversion=request.risk_aversion
                )
            elif request.method == OptimizationMethod.RISK_PARITY:
                weights = self.risk_parity_optimization(covariance)
            elif request.method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = self.minimum_variance_optimization(covariance)
            elif request.method == OptimizationMethod.MAXIMUM_SHARPE:
                weights = self.maximum_sharpe_optimization(expected_returns, covariance)
            else:
                raise ValueError(f"Unknown optimization method: {request.method}")
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(weights.T @ covariance @ weights)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Assume $1M portfolio for allocation calculation
            total_value = 1000000
            
            # Create allocations
            allocations = []
            for i, asset in enumerate(request.assets):
                if weights[i] > 0.001:  # Only include significant allocations
                    market_value = weights[i] * total_value
                    shares = int(market_value / asset.current_price)
                    
                    allocation = PortfolioAllocation(
                        symbol=asset.symbol,
                        weight=weights[i],
                        shares=shares,
                        market_value=market_value
                    )
                    allocations.append(allocation)
            
            result = OptimizationResult(
                portfolio_id=request.portfolio_id,
                method=request.method,
                allocations=allocations,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                total_value=total_value,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global variables
portfolio_optimizer = None
db_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global portfolio_optimizer, db_engine
    
    # Startup
    logger.info("Starting Portfolio Service...")
    
    # Initialize database connection
    db_engine = create_engine("postgresql://finsim:finsim123@postgres:5432/finsim")
    portfolio_optimizer = PortfolioOptimizer(db_engine)
    
    logger.info("Portfolio Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Portfolio Service...")
    if db_engine:
        db_engine.dispose()

app = FastAPI(
    title="FinSim Portfolio Service",
    description="Portfolio optimization and efficient frontier calculation",
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

@app.post("/api/v1/portfolio/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    """Optimize portfolio allocation"""
    try:
        result = await portfolio_optimizer.optimize_portfolio(request)
        return result.model_dump()
    
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/portfolio/efficient-frontier")
async def calculate_efficient_frontier(request: OptimizationRequest):
    """Calculate efficient frontier"""
    try:
        symbols = [asset.symbol for asset in request.assets]
        expected_returns, covariance = portfolio_optimizer.estimate_returns_covariance(symbols)
        
        frontier_points = portfolio_optimizer.calculate_efficient_frontier(
            expected_returns, covariance, symbols
        )
        
        # Find optimal portfolio (max Sharpe)
        optimal_point = max(frontier_points, key=lambda p: p.sharpe_ratio)
        
        result = EfficientFrontier(
            portfolio_id=request.portfolio_id,
            points=frontier_points,
            optimal_portfolio=optimal_point,
            timestamp=datetime.now()
        )
        
        return result.model_dump()
    
    except Exception as e:
        logger.error(f"Error calculating efficient frontier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/portfolio/{portfolio_id}/rebalance")
async def suggest_rebalancing(portfolio_id: str, 
                            frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY):
    """Suggest portfolio rebalancing"""
    try:
        # In production, this would analyze current positions vs target allocation
        # For demo, we'll simulate rebalancing suggestions
        
        suggestions = [
            {
                "symbol": "AAPL",
                "current_weight": 0.25,
                "target_weight": 0.20,
                "action": "REDUCE",
                "amount": 50000
            },
            {
                "symbol": "GOOGL",
                "current_weight": 0.15,
                "target_weight": 0.20,
                "action": "INCREASE",
                "amount": 25000
            },
            {
                "symbol": "MSFT",
                "current_weight": 0.30,
                "target_weight": 0.25,
                "action": "REDUCE",
                "amount": 25000
            }
        ]
        
        return {
            "portfolio_id": portfolio_id,
            "rebalancing_frequency": frequency,
            "suggestions": suggestions,
            "total_turnover": 100000,
            "estimated_cost": 500,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error suggesting rebalancing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/portfolio/{portfolio_id}/risk-attribution")
async def risk_attribution_analysis(portfolio_id: str):
    """Portfolio risk attribution analysis"""
    try:
        # Simulate risk attribution by factors
        risk_factors = {
            "market": 0.65,
            "size": 0.15,
            "value": 0.08,
            "momentum": 0.05,
            "quality": 0.04,
            "specific": 0.03
        }
        
        return {
            "portfolio_id": portfolio_id,
            "risk_attribution": risk_factors,
            "total_risk": 0.18,  # 18% annualized volatility
            "systematic_risk": 0.175,
            "idiosyncratic_risk": 0.005,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error in risk attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )