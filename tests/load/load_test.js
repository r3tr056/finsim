import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
export let errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.05'],   // Error rate must be below 5%
    errors: ['rate<0.1'],             // Custom error rate must be below 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'];

export default function () {
  // Test suite for FinSim platform load testing
  
  // 1. Market Data API Load Test
  testMarketDataAPI();
  
  // 2. Agents API Load Test
  testAgentsAPI();
  
  // 3. Simulation Engine Load Test
  testSimulationEngine();
  
  // 4. WebSocket Load Test
  testWebSocketConnection();
  
  sleep(1);
}

function testMarketDataAPI() {
  let symbol = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)];
  
  // Test quotes endpoint
  let quotesResponse = http.get(`${BASE_URL}/api/v1/quotes/${symbol}`, {
    timeout: '10s',
  });
  
  let quotesSuccess = check(quotesResponse, {
    'quotes status is 200': (r) => r.status === 200,
    'quotes response time < 500ms': (r) => r.timings.duration < 500,
    'quotes has price data': (r) => {
      try {
        let data = JSON.parse(r.body);
        return data.price !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!quotesSuccess) {
    errorRate.add(1);
  }
  
  // Test historical data endpoint
  let historicalResponse = http.get(`${BASE_URL}/api/v1/historical/${symbol}?period=5d`, {
    timeout: '15s',
  });
  
  let historicalSuccess = check(historicalResponse, {
    'historical status is 200': (r) => r.status === 200,
    'historical response time < 1s': (r) => r.timings.duration < 1000,
    'historical has data array': (r) => {
      try {
        let data = JSON.parse(r.body);
        return Array.isArray(data) && data.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!historicalSuccess) {
    errorRate.add(1);
  }
}

function testAgentsAPI() {
  // Test agent creation
  let agentConfig = {
    agent_id: `load_test_agent_${__VU}_${__ITER}`,
    agent_type: 'heuristic',
    strategy: 'momentum',
    symbols: [SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)]],
    parameters: {
      rsi_period: 14,
      rsi_overbought: 70,
      rsi_oversold: 30
    },
    enabled: true
  };
  
  let createResponse = http.post(`${BASE_URL}:8001/api/v1/agents`, JSON.stringify(agentConfig), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '10s',
  });
  
  let createSuccess = check(createResponse, {
    'agent creation status is 200': (r) => r.status === 200,
    'agent creation response time < 1s': (r) => r.timings.duration < 1000,
    'agent creation returns agent_id': (r) => {
      try {
        let data = JSON.parse(r.body);
        return data.agent_id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!createSuccess) {
    errorRate.add(1);
    return;
  }
  
  let agentId = agentConfig.agent_id;
  
  // Test agent status retrieval
  let statusResponse = http.get(`${BASE_URL}:8001/api/v1/agents/${agentId}`, {
    timeout: '5s',
  });
  
  let statusSuccess = check(statusResponse, {
    'agent status is 200': (r) => r.status === 200,
    'agent status response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  if (!statusSuccess) {
    errorRate.add(1);
  }
  
  // Test agent deletion (cleanup)
  let deleteResponse = http.del(`${BASE_URL}:8001/api/v1/agents/${agentId}`, null, {
    timeout: '5s',
  });
  
  check(deleteResponse, {
    'agent deletion status is 200': (r) => r.status === 200,
  });
}

function testSimulationEngine() {
  // Test order placement
  let orderData = {
    symbol: SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)],
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    order_type: 'market',
    quantity: Math.floor(Math.random() * 100) + 1,
    agent_id: `load_test_agent_${__VU}`
  };
  
  let orderResponse = http.post(`${BASE_URL}:8002/api/v1/orders`, JSON.stringify(orderData), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '10s',
  });
  
  let orderSuccess = check(orderResponse, {
    'order placement status is 200 or 201': (r) => r.status === 200 || r.status === 201,
    'order placement response time < 1s': (r) => r.timings.duration < 1000,
    'order placement returns order_id': (r) => {
      try {
        let data = JSON.parse(r.body);
        return data.order_id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!orderSuccess) {
    errorRate.add(1);
    return;
  }
  
  // Test orderbook retrieval
  let orderbookResponse = http.get(`${BASE_URL}:8002/api/v1/orderbook/${orderData.symbol}`, {
    timeout: '5s',
  });
  
  let orderbookSuccess = check(orderbookResponse, {
    'orderbook status is 200': (r) => r.status === 200,
    'orderbook response time < 500ms': (r) => r.timings.duration < 500,
    'orderbook has bids and asks': (r) => {
      try {
        let data = JSON.parse(r.body);
        return data.bids !== undefined && data.asks !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!orderbookSuccess) {
    errorRate.add(1);
  }
}

function testWebSocketConnection() {
  const url = `ws://localhost:8000/ws/market`;
  
  const response = ws.connect(url, null, function (socket) {
    socket.on('open', function open() {
      console.log('WebSocket connection established');
      
      // Subscribe to market data
      socket.send(JSON.stringify({
        action: 'subscribe',
        symbols: ['AAPL', 'GOOGL']
      }));
    });
    
    socket.on('message', function message(data) {
      let success = check(data, {
        'websocket message is valid JSON': (d) => {
          try {
            JSON.parse(d);
            return true;
          } catch (e) {
            return false;
          }
        },
        'websocket message has required fields': (d) => {
          try {
            let parsed = JSON.parse(d);
            return parsed.symbol !== undefined && parsed.price !== undefined;
          } catch (e) {
            return false;
          }
        },
      });
      
      if (!success) {
        errorRate.add(1);
      }
    });
    
    socket.on('error', function error(e) {
      console.log('WebSocket error:', e);
      errorRate.add(1);
    });
    
    // Keep connection open for a short time
    sleep(2);
    socket.close();
  });
  
  check(response, {
    'websocket connection successful': (r) => r && r.status === 200,
  });
}

export function setup() {
  // Setup phase - run once before load test
  console.log('Starting FinSim load test setup...');
  
  // Health check all services
  let services = [
    { name: 'Market Data', url: `${BASE_URL}/health` },
    { name: 'Agents', url: `${BASE_URL}:8001/health` },
    { name: 'Simulation', url: `${BASE_URL}:8002/health` },
    { name: 'Risk Analytics', url: `${BASE_URL}:8003/health` },
    { name: 'Portfolio', url: `${BASE_URL}:8004/health` },
  ];
  
  let availableServices = [];
  
  for (let service of services) {
    try {
      let response = http.get(service.url, { timeout: '5s' });
      if (response.status === 200) {
        availableServices.push(service.name);
        console.log(`✓ ${service.name} service is available`);
      } else {
        console.log(`⚠ ${service.name} service returned status ${response.status}`);
      }
    } catch (e) {
      console.log(`✗ ${service.name} service is not available: ${e}`);
    }
  }
  
  console.log(`Setup complete. ${availableServices.length}/${services.length} services available.`);
  
  return { availableServices: availableServices };
}

export function teardown(data) {
  // Teardown phase - run once after load test
  console.log('FinSim load test teardown...');
  console.log(`Test completed with ${data.availableServices.length} services tested.`);
  
  // Cleanup any test data if needed
  console.log('Cleanup completed.');
}

// Scenario-specific load tests
export let scenarios = {
  market_data_heavy: {
    executor: 'constant-vus',
    vus: 50,
    duration: '5m',
    exec: 'marketDataScenario',
  },
  trading_simulation: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 20 },
      { duration: '3m', target: 20 },
      { duration: '1m', target: 0 },
    ],
    exec: 'tradingScenario',
  },
  websocket_stress: {
    executor: 'constant-vus',
    vus: 100,
    duration: '2m',
    exec: 'websocketScenario',
  },
};

export function marketDataScenario() {
  // Heavy market data requests
  for (let i = 0; i < 5; i++) {
    testMarketDataAPI();
    sleep(0.1);
  }
}

export function tradingScenario() {
  // Trading simulation focused test
  testAgentsAPI();
  testSimulationEngine();
  sleep(1);
}

export function websocketScenario() {
  // WebSocket focused test
  testWebSocketConnection();
}