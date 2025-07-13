"""
Unit tests for the Market Data Service
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Import the market data service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../services/market-data'))

from app.main import app, MarketDataService, YFinanceConnector


class TestMarketDataService:
    """Test suite for Market Data Service"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
        self.service = MarketDataService()
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
    def test_list_symbols_endpoint(self):
        """Test symbols listing endpoint"""
        response = self.client.get("/api/v1/symbols")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
    @patch('yfinance.Ticker')
    def test_quote_endpoint_success(self, mock_ticker):
        """Test successful quote retrieval"""
        # Mock yfinance response
        mock_info = {
            'regularMarketPrice': 150.25,
            'regularMarketOpen': 149.50,
            'regularMarketDayHigh': 151.00,
            'regularMarketDayLow': 148.75,
            'regularMarketVolume': 1000000,
            'marketCap': 2500000000000
        }
        mock_ticker.return_value.info = mock_info
        
        response = self.client.get("/api/v1/quotes/AAPL")
        assert response.status_code == 200
        data = response.json()
        
        assert data["symbol"] == "AAPL"
        assert data["price"] == 150.25
        assert data["open"] == 149.50
        assert data["high"] == 151.00
        assert data["low"] == 148.75
        assert data["volume"] == 1000000
        
    def test_quote_endpoint_invalid_symbol(self):
        """Test quote endpoint with invalid symbol"""
        response = self.client.get("/api/v1/quotes/INVALID")
        assert response.status_code == 404
        
    @patch('yfinance.Ticker')
    def test_historical_data_endpoint(self, mock_ticker):
        """Test historical data retrieval"""
        # Mock historical data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        mock_hist = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [104, 105, 106, 107, 108],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        response = self.client.get("/api/v1/historical/AAPL")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 5
        assert all(key in data[0] for key in ['date', 'open', 'high', 'low', 'close', 'volume'])
        
    def test_historical_data_with_parameters(self):
        """Test historical data with custom parameters"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_hist = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [99], 
                'Close': [104], 'Volume': [1000000]
            }, index=[pd.Timestamp('2023-01-01')])
            
            mock_ticker.return_value.history.return_value = mock_hist
            
            response = self.client.get("/api/v1/historical/AAPL?period=1mo&interval=1d")
            assert response.status_code == 200
            
            # Verify the ticker.history was called with correct parameters
            mock_ticker.return_value.history.assert_called_with(period='1mo', interval='1d')


class TestYFinanceConnector:
    """Test suite for YFinance Connector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.connector = YFinanceConnector()
        
    @patch('yfinance.Ticker')
    def test_get_quote_success(self, mock_ticker):
        """Test successful quote retrieval"""
        mock_info = {
            'regularMarketPrice': 150.25,
            'regularMarketOpen': 149.50,
            'regularMarketDayHigh': 151.00,
            'regularMarketDayLow': 148.75,
            'regularMarketVolume': 1000000
        }
        mock_ticker.return_value.info = mock_info
        
        quote = self.connector.get_quote("AAPL")
        
        assert quote is not None
        assert quote['symbol'] == 'AAPL'
        assert quote['price'] == 150.25
        assert quote['volume'] == 1000000
        
    @patch('yfinance.Ticker')
    def test_get_quote_failure(self, mock_ticker):
        """Test quote retrieval failure"""
        mock_ticker.return_value.info = {}
        
        quote = self.connector.get_quote("INVALID")
        assert quote is None
        
    @patch('yfinance.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval"""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        mock_hist = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        data = self.connector.get_historical_data("AAPL", period="5d")
        
        assert len(data) == 3
        assert all(isinstance(item, dict) for item in data)
        assert all('date' in item for item in data)
        
    @patch('yfinance.Ticker')
    def test_get_historical_data_empty(self, mock_ticker):
        """Test historical data retrieval with empty response"""
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        data = self.connector.get_historical_data("INVALID")
        assert data == []


class TestKafkaIntegration:
    """Test suite for Kafka integration"""
    
    @patch('kafka.KafkaProducer')
    def test_kafka_producer_initialization(self, mock_producer):
        """Test Kafka producer initialization"""
        from app.main import kafka_producer
        assert mock_producer.called
        
    @patch('kafka.KafkaProducer')
    async def test_publish_market_data(self, mock_producer):
        """Test publishing market data to Kafka"""
        mock_instance = Mock()
        mock_producer.return_value = mock_instance
        
        service = MarketDataService()
        service.kafka_producer = mock_instance
        
        test_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
        await service.publish_market_data(test_data)
        
        # Verify that send was called
        mock_instance.send.assert_called_once()
        call_args = mock_instance.send.call_args
        assert call_args[0][0] == 'market-data'  # topic name


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_symbol_validation(self):
        """Test symbol validation"""
        client = TestClient(app)
        
        # Valid symbol
        response = client.get("/api/v1/quotes/AAPL")
        assert response.status_code in [200, 404]  # 404 is ok if mock fails
        
        # Invalid symbol format
        response = client.get("/api/v1/quotes/INVALID123!")
        assert response.status_code == 422  # Validation error
        
    def test_period_validation(self):
        """Test period parameter validation"""
        client = TestClient(app)
        
        # Valid period
        response = client.get("/api/v1/historical/AAPL?period=1mo")
        assert response.status_code in [200, 404]
        
        # Invalid period
        response = client.get("/api/v1/historical/AAPL?period=invalid")
        assert response.status_code == 422


class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_service_unavailable_handling(self):
        """Test handling of external service unavailability"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("Service unavailable")
            
            client = TestClient(app)
            response = client.get("/api/v1/quotes/AAPL")
            
            # Should handle the error gracefully
            assert response.status_code == 500
            
    def test_network_timeout_handling(self):
        """Test handling of network timeouts"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = TimeoutError("Request timeout")
            
            client = TestClient(app)
            response = client.get("/api/v1/quotes/AAPL")
            
            assert response.status_code == 500


class TestPerformance:
    """Test suite for performance characteristics"""
    
    @patch('yfinance.Ticker')
    def test_response_time(self, mock_ticker):
        """Test API response time"""
        import time
        
        mock_info = {'regularMarketPrice': 150.25}
        mock_ticker.return_value.info = mock_info
        
        client = TestClient(app)
        
        start_time = time.time()
        response = client.get("/api/v1/quotes/AAPL")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200
        
    @patch('yfinance.Ticker')
    def test_concurrent_requests(self, mock_ticker):
        """Test handling of concurrent requests"""
        import threading
        
        mock_info = {'regularMarketPrice': 150.25}
        mock_ticker.return_value.info = mock_info
        
        client = TestClient(app)
        results = []
        
        def make_request():
            response = client.get("/api/v1/quotes/AAPL")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__])