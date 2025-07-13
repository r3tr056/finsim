"""
FinSim Shared Libraries
Provides common utilities and data structures used across all services
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import logging


# Configure logging
logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class AgentType(str, Enum):
    """Agent type enumeration"""
    HEURISTIC = "heuristic"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRU = "gru"
    DQN = "dqn"
    PPO = "ppo"
    A3C = "a3c"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    agent_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary"""
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Trade:
    """Trade execution data structure"""
    trade_id: str
    symbol: str
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    timestamp: Optional[datetime] = None
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.trade_id is None:
            self.trade_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return result


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    timestamp: Optional[datetime] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    volume: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return result


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return asdict(self)


@dataclass
class Portfolio:
    """Portfolio data structure"""
    portfolio_id: str
    cash: float
    positions: Dict[str, Position]
    total_value: float
    total_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary"""
        result = asdict(self)
        result['positions'] = {k: v.to_dict() for k, v in self.positions.items()}
        return result


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_order(order_data: Dict[str, Any]) -> bool:
        """Validate order data"""
        required_fields = ['symbol', 'side', 'order_type', 'quantity']
        
        for field in required_fields:
            if field not in order_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate order side
        if order_data['side'] not in [side.value for side in OrderSide]:
            logger.error(f"Invalid order side: {order_data['side']}")
            return False
        
        # Validate order type
        if order_data['order_type'] not in [ot.value for ot in OrderType]:
            logger.error(f"Invalid order type: {order_data['order_type']}")
            return False
        
        # Validate quantity
        if not isinstance(order_data['quantity'], (int, float)) or order_data['quantity'] <= 0:
            logger.error(f"Invalid quantity: {order_data['quantity']}")
            return False
        
        # Validate price for limit orders
        if order_data['order_type'] in ['limit', 'stop_limit']:
            if 'price' not in order_data or not isinstance(order_data['price'], (int, float)) or order_data['price'] <= 0:
                logger.error(f"Invalid price for limit order: {order_data.get('price')}")
                return False
        
        return True
    
    @staticmethod
    def validate_market_data(data: Dict[str, Any]) -> bool:
        """Validate market data"""
        required_fields = ['symbol', 'price']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate price
        if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
            logger.error(f"Invalid price: {data['price']}")
            return False
        
        return True


class MessageBus:
    """Simple message bus for inter-service communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
    
    def subscribe(self, topic: str, callback: callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    def publish(self, topic: str, message: Any):
        """Publish message to topic"""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
    
    def unsubscribe(self, topic: str, callback: callable):
        """Unsubscribe from topic"""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)


class TimeSeriesData:
    """Time series data handling utilities"""
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def add_data_point(self, symbol: str, timestamp: datetime, **kwargs):
        """Add a data point to time series"""
        if symbol not in self.data:
            self.data[symbol] = pd.DataFrame()
        
        new_row = pd.DataFrame([{'timestamp': timestamp, **kwargs}])
        new_row.set_index('timestamp', inplace=True)
        
        if self.data[symbol].empty:
            self.data[symbol] = new_row
        else:
            self.data[symbol] = pd.concat([self.data[symbol], new_row])
    
    def get_latest(self, symbol: str, n: int = 1) -> Optional[pd.DataFrame]:
        """Get latest n data points"""
        if symbol not in self.data or self.data[symbol].empty:
            return None
        return self.data[symbol].tail(n)
    
    def get_range(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get data within time range"""
        if symbol not in self.data or self.data[symbol].empty:
            return None
        return self.data[symbol][start_time:end_time]
    
    def calculate_returns(self, symbol: str, price_column: str = 'price') -> Optional[pd.Series]:
        """Calculate returns for a symbol"""
        if symbol not in self.data or self.data[symbol].empty:
            return None
        
        prices = self.data[symbol][price_column]
        return prices.pct_change().dropna()
    
    def calculate_volatility(self, symbol: str, window: int = 20, price_column: str = 'price') -> Optional[pd.Series]:
        """Calculate rolling volatility"""
        returns = self.calculate_returns(symbol, price_column)
        if returns is None:
            return None
        
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


class RiskMetrics:
    """Risk calculation utilities"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if returns.empty:
            return 0.0
        
        alpha = 1 - confidence_level
        return np.percentile(returns, alpha * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if tail_returns.empty:
            return var
        
        return tail_returns.mean()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if cumulative_returns.empty:
            return 0.0
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()


class PerformanceTracker:
    """Performance tracking utilities"""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.cash = 100000.0  # Default starting cash
        self.initial_value = self.cash
    
    def add_trade(self, trade: Trade):
        """Add trade to tracking"""
        self.trades.append(trade)
        self.update_positions(trade)
    
    def update_positions(self, trade: Trade):
        """Update positions based on trade"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0
            )
        
        position = self.positions[symbol]
        
        # Update position based on trade
        if trade.buyer_id == 'self':  # We bought
            old_quantity = position.quantity
            old_value = old_quantity * position.avg_price
            new_quantity = old_quantity + trade.quantity
            new_value = old_value + trade.quantity * trade.price
            
            position.quantity = new_quantity
            position.avg_price = new_value / new_quantity if new_quantity > 0 else 0.0
            self.cash -= trade.quantity * trade.price
            
        elif trade.seller_id == 'self':  # We sold
            position.quantity -= trade.quantity
            if position.quantity < 0:
                position.quantity = 0.0  # Prevent negative positions
            self.cash += trade.quantity * trade.price
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.market_value = position.quantity * current_prices[symbol]
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
                total_value += position.market_value
        
        return total_value
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        portfolio_value = self.calculate_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_value) / self.initial_value
        
        # Calculate returns series for additional metrics
        trade_returns = []
        for trade in self.trades:
            # Simplified return calculation
            trade_returns.append(trade.price / 100.0 - 1.0)  # Assuming base price of 100
        
        returns_series = pd.Series(trade_returns)
        
        return {
            'total_return': total_return,
            'portfolio_value': portfolio_value,
            'total_trades': len(self.trades),
            'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns_series),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(returns_series.cumsum()),
            'var_95': RiskMetrics.calculate_var(returns_series),
            'expected_shortfall_95': RiskMetrics.calculate_expected_shortfall(returns_series)
        }


class ConfigurationManager:
    """Configuration management utilities"""
    
    DEFAULT_CONFIG = {
        'market_data': {
            'update_frequency': 1.0,  # seconds
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
            'price_volatility': 0.02
        },
        'simulation': {
            'tick_size': 0.01,
            'commission': 0.001,
            'slippage': 0.0005
        },
        'agents': {
            'max_position_size': 1000,
            'risk_tolerance': 0.1
        },
        'database': {
            'connection_string': 'postgresql://user:pass@localhost:5432/finsim',
            'pool_size': 10
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'topics': {
                'market_data': 'market-data',
                'trades': 'trades',
                'orders': 'orders'
            }
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    self.config.update(custom_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


# Global instances
message_bus = MessageBus()
time_series_data = TimeSeriesData()
config_manager = ConfigurationManager()


def get_logger(name: str) -> logging.Logger:
    """Get configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2%}"


def generate_id() -> str:
    """Generate unique ID"""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)


# Export main classes and functions
__all__ = [
    'Order', 'Trade', 'MarketData', 'Position', 'Portfolio',
    'OrderSide', 'OrderType', 'OrderStatus', 'AgentType',
    'DataValidator', 'MessageBus', 'TimeSeriesData', 'RiskMetrics',
    'PerformanceTracker', 'ConfigurationManager',
    'message_bus', 'time_series_data', 'config_manager',
    'get_logger', 'format_currency', 'format_percentage', 'generate_id', 'utc_now'
]