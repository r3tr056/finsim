"""
Unit tests for the Agents Service
"""

import pytest
import asyncio
import json
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# Import the agents service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../services/agents'))

from app.main import app, BaseAgent, MomentumAgent, LSTMAgent, DQNAgent, PPOAgent


class TestBaseAgent:
    """Test suite for Base Agent class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.HEURISTIC,
            symbols=["AAPL", "GOOGL"],
            parameters={"test_param": 1.0}
        )
        self.agent = BaseAgent(self.config)
        
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.symbols == ["AAPL", "GOOGL"]
        assert self.agent.enabled == True
        assert len(self.agent.positions) == 2
        assert all(pos == 0 for pos in self.agent.positions.values())
        
    async def test_agent_start_stop(self):
        """Test agent start/stop functionality"""
        assert not self.agent.running
        
        await self.agent.start()
        assert self.agent.running
        
        await self.agent.stop()
        assert not self.agent.running
        
    async def test_update_market_data(self):
        """Test market data update"""
        data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
        await self.agent.update_market_data(data)
        assert self.agent.last_prices['AAPL'] == 150.25
        
    async def test_update_market_data_unknown_symbol(self):
        """Test market data update for unknown symbol"""
        data = {
            'symbol': 'UNKNOWN',
            'price': 100.0,
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
        await self.agent.update_market_data(data)
        assert 'UNKNOWN' not in self.agent.last_prices
        
    @patch('httpx.AsyncClient')
    async def test_place_order_success(self, mock_client):
        """Test successful order placement"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"order_id": "12345", "status": "filled"}
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await self.agent.place_order("AAPL", "buy", 100, 150.0)
        
        assert result is not None
        assert result["order_id"] == "12345"
        assert self.agent.performance["total_trades"] == 1
        
    @patch('httpx.AsyncClient')
    async def test_place_order_failure(self, mock_client):
        """Test failed order placement"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid order"
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await self.agent.place_order("AAPL", "buy", 100, 150.0)
        assert result is None


class TestMomentumAgent:
    """Test suite for Momentum Agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType, Strategy
        self.config = AgentConfig(
            agent_id="momentum_agent",
            agent_type=AgentType.HEURISTIC,
            strategy=Strategy.MOMENTUM,
            symbols=["AAPL"],
            parameters={"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30}
        )
        self.agent = MomentumAgent(self.config)
        
    def test_momentum_agent_initialization(self):
        """Test momentum agent initialization"""
        assert self.agent.rsi_period == 14
        assert self.agent.rsi_overbought == 70
        assert self.agent.rsi_oversold == 30
        assert len(self.agent.price_history) == 1
        
    @patch('talib.RSI')
    @patch.object(MomentumAgent, 'place_order')
    async def test_buy_signal(self, mock_place_order, mock_rsi):
        """Test buy signal generation"""
        # Setup price history
        self.agent.price_history['AAPL'] = [100 + i for i in range(20)]
        
        # Mock RSI to return oversold value
        mock_rsi.return_value = np.array([25.0])  # Oversold
        mock_place_order.return_value = {"order_id": "123"}
        
        data = {'price': 120.0}
        await self.agent.make_decision('AAPL', data)
        
        mock_place_order.assert_called_once_with('AAPL', 'buy', 100, 120.0 * 1.001)
        assert self.agent.positions['AAPL'] == 100
        
    @patch('talib.RSI')
    @patch.object(MomentumAgent, 'place_order')
    async def test_sell_signal(self, mock_place_order, mock_rsi):
        """Test sell signal generation"""
        # Setup price history and position
        self.agent.price_history['AAPL'] = [100 + i for i in range(20)]
        self.agent.positions['AAPL'] = 100
        
        # Mock RSI to return overbought value
        mock_rsi.return_value = np.array([75.0])  # Overbought
        mock_place_order.return_value = {"order_id": "123"}
        
        data = {'price': 120.0}
        await self.agent.make_decision('AAPL', data)
        
        mock_place_order.assert_called_once_with('AAPL', 'sell', 100, 120.0 * 0.999)
        assert self.agent.positions['AAPL'] == 0
        
    async def test_insufficient_price_history(self):
        """Test behavior with insufficient price history"""
        # Only add a few prices
        self.agent.price_history['AAPL'] = [100, 101, 102]
        
        with patch.object(self.agent, 'place_order') as mock_place_order:
            data = {'price': 103.0}
            await self.agent.make_decision('AAPL', data)
            
            # Should not place any orders
            mock_place_order.assert_not_called()


class TestLSTMAgent:
    """Test suite for LSTM Agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="lstm_agent",
            agent_type=AgentType.LSTM,
            symbols=["AAPL"],
            parameters={"sequence_length": 20}
        )
        self.agent = LSTMAgent(self.config)
        
    def test_lstm_agent_initialization(self):
        """Test LSTM agent initialization"""
        assert self.agent.sequence_length == 20
        assert hasattr(self.agent, 'model')
        assert hasattr(self.agent, 'scaler')
        assert not self.agent.is_trained
        
    def test_prepare_features_insufficient_data(self):
        """Test feature preparation with insufficient data"""
        self.agent.price_history['AAPL'] = [100, 101, 102]
        features = self.agent.prepare_features('AAPL')
        assert features is None
        
    @patch('talib.SMA')
    @patch('talib.RSI')
    def test_prepare_features_success(self, mock_rsi, mock_sma):
        """Test successful feature preparation"""
        # Setup sufficient price history
        prices = [100 + np.random.randn() for _ in range(50)]
        self.agent.price_history['AAPL'] = prices
        
        # Mock technical indicators
        mock_sma.return_value = np.array([100] * 30)
        mock_rsi.return_value = np.array([50] * 30)
        
        features = self.agent.prepare_features('AAPL')
        
        assert features is not None
        assert features.shape[1] == 5  # 5 features per observation
        
    @patch.object(LSTMAgent, 'prepare_features')
    @patch.object(LSTMAgent, 'place_order')
    async def test_make_decision_training(self, mock_place_order, mock_prepare_features):
        """Test decision making and training"""
        mock_features = np.random.randn(20, 5)
        mock_prepare_features.return_value = mock_features
        mock_place_order.return_value = {"order_id": "123"}
        
        data = {'price': 150.0}
        await self.agent.make_decision('AAPL', data)
        
        # Should mark as trained after first decision
        assert self.agent.is_trained


class TestDQNAgent:
    """Test suite for DQN Agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="dqn_agent",
            agent_type=AgentType.DQN,
            symbols=["AAPL"],
            parameters={"epsilon": 0.1}
        )
        self.agent = DQNAgent(self.config)
        
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization"""
        assert self.agent.state_size == 10
        assert self.agent.action_size == 3
        assert self.agent.epsilon == 0.1
        assert hasattr(self.agent, 'model')
        
    def test_get_state_insufficient_data(self):
        """Test state generation with insufficient data"""
        self.agent.price_history['AAPL'] = [100, 101]
        state = self.agent.get_state('AAPL')
        assert state is None
        
    def test_get_state_success(self):
        """Test successful state generation"""
        # Setup sufficient price history
        self.agent.price_history['AAPL'] = [100 + i * 0.1 for i in range(25)]
        state = self.agent.get_state('AAPL')
        
        assert state is not None
        assert len(state) == 10
        assert isinstance(state, np.ndarray)
        
    @patch.object(DQNAgent, 'get_state')
    @patch.object(DQNAgent, 'place_order')
    @patch('torch.no_grad')
    async def test_make_decision_with_model(self, mock_no_grad, mock_place_order, mock_get_state):
        """Test decision making with DQN model"""
        mock_state = np.random.randn(10)
        mock_get_state.return_value = mock_state
        mock_place_order.return_value = {"order_id": "123"}
        
        # Mock torch operations
        mock_tensor = Mock()
        mock_q_values = Mock()
        mock_q_values.argmax.return_value.item.return_value = 1  # Buy action
        
        with patch('torch.FloatTensor', return_value=mock_tensor):
            with patch.object(self.agent.model, 'forward', return_value=mock_q_values):
                data = {'price': 150.0}
                await self.agent.make_decision('AAPL', data)
                
                mock_place_order.assert_called_once()


class TestPPOAgent:
    """Test suite for PPO Agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="ppo_agent",
            agent_type=AgentType.PPO,
            symbols=["AAPL"],
            parameters={"epsilon": 0.2, "gamma": 0.99}
        )
        self.agent = PPOAgent(self.config)
        
    def test_ppo_agent_initialization(self):
        """Test PPO agent initialization"""
        assert self.agent.state_size == 10
        assert self.agent.action_size == 3
        assert self.agent.epsilon == 0.2
        assert hasattr(self.agent, 'model')
        assert hasattr(self.agent, 'optimizer')
        
    def test_get_state_success(self):
        """Test successful state generation"""
        self.agent.price_history['AAPL'] = [100 + i * 0.1 for i in range(25)]
        state = self.agent.get_state('AAPL')
        
        assert state is not None
        assert len(state) == 10


class TestAgentsAPI:
    """Test suite for Agents API endpoints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    def test_create_agent_momentum(self):
        """Test creating a momentum agent"""
        agent_config = {
            "agent_id": "test_momentum",
            "agent_type": "heuristic",
            "strategy": "momentum",
            "symbols": ["AAPL"],
            "parameters": {"rsi_period": 14},
            "enabled": True
        }
        
        response = self.client.post("/api/v1/agents", json=agent_config)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["agent_id"] == "test_momentum"
        
    def test_create_agent_lstm(self):
        """Test creating an LSTM agent"""
        agent_config = {
            "agent_id": "test_lstm",
            "agent_type": "lstm",
            "symbols": ["AAPL"],
            "parameters": {"sequence_length": 20},
            "enabled": True
        }
        
        response = self.client.post("/api/v1/agents", json=agent_config)
        assert response.status_code == 200
        
    def test_create_agent_duplicate_id(self):
        """Test creating agent with duplicate ID"""
        agent_config = {
            "agent_id": "duplicate_test",
            "agent_type": "dqn",
            "symbols": ["AAPL"],
            "enabled": True
        }
        
        # Create first agent
        response1 = self.client.post("/api/v1/agents", json=agent_config)
        assert response1.status_code == 200
        
        # Try to create duplicate
        response2 = self.client.post("/api/v1/agents", json=agent_config)
        assert response2.status_code == 400
        
    def test_list_agents_empty(self):
        """Test listing agents when none exist"""
        response = self.client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    def test_get_agent_not_found(self):
        """Test getting non-existent agent"""
        response = self.client.get("/api/v1/agents/nonexistent")
        assert response.status_code == 404
        
    def test_agent_start_stop_not_found(self):
        """Test starting/stopping non-existent agent"""
        response = self.client.post("/api/v1/agents/nonexistent/start")
        assert response.status_code == 404
        
        response = self.client.post("/api/v1/agents/nonexistent/stop")
        assert response.status_code == 404
        
    def test_delete_agent_not_found(self):
        """Test deleting non-existent agent"""
        response = self.client.delete("/api/v1/agents/nonexistent")
        assert response.status_code == 404


class TestAgentPerformance:
    """Test suite for agent performance tracking"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="perf_test_agent",
            agent_type=AgentType.HEURISTIC,
            symbols=["AAPL"]
        )
        self.agent = BaseAgent(self.config)
        
    def test_initial_performance_metrics(self):
        """Test initial performance metrics"""
        assert self.agent.performance["total_pnl"] == 0.0
        assert self.agent.performance["total_trades"] == 0
        assert self.agent.performance["win_rate"] == 0.0
        assert self.agent.performance["sharpe_ratio"] == 0.0
        
    @patch.object(BaseAgent, 'place_order')
    async def test_performance_tracking_after_trade(self, mock_place_order):
        """Test performance tracking after trade"""
        mock_place_order.return_value = {"order_id": "123"}
        
        await self.agent.place_order("AAPL", "buy", 100, 150.0)
        
        assert self.agent.performance["total_trades"] == 1


class TestAgentMemoryManagement:
    """Test suite for agent memory management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from app.main import AgentConfig, AgentType
        self.config = AgentConfig(
            agent_id="memory_test_agent",
            agent_type=AgentType.LSTM,
            symbols=["AAPL"]
        )
        self.agent = LSTMAgent(self.config)
        
    async def test_price_history_limit(self):
        """Test price history memory limit"""
        # Add many prices
        for i in range(300):
            data = {'price': 100 + i}
            await self.agent.update_market_data({'symbol': 'AAPL', **data})
        
        # Should be limited to 200
        assert len(self.agent.price_history['AAPL']) <= 200


if __name__ == "__main__":
    pytest.main([__file__])