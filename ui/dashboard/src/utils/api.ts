import axios, { AxiosInstance, AxiosResponse } from 'axios'
import { 
  QuoteData, 
  OrderBookSnapshot, 
  Trade, 
  Agent, 
  AgentConfig, 
  RiskMetrics, 
  OptimizationResult,
  User,
  ApiResponse 
} from '../types'

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authAPI = {
  login: (credentials: { email: string; password: string }) =>
    api.post<{ access_token: string; refresh_token: string; token_type: string; expires_in: number }>('/v1/auth/login', credentials),
  
  register: (userData: { email: string; password: string; fullName: string; role?: string }) =>
    api.post<{ message: string; user_id: string }>('/v1/auth/register', userData),
  
  getCurrentUser: () =>
    api.get<User>('/v1/auth/me'),
  
  getUserPermissions: () =>
    api.get<{ user_id: string; permissions: string[] }>('/v1/auth/permissions'),
  
  refreshToken: (refreshToken: string) =>
    api.post<{ access_token: string; refresh_token: string; token_type: string; expires_in: number }>('/v1/auth/refresh', { refresh_token: refreshToken }),
}

// Market Data API
export const marketAPI = {
  getQuote: (symbol: string) =>
    api.get<QuoteData>(`/v1/quotes/${symbol}`),
  
  getHistoricalData: (symbols: string[], interval: string = '1d') =>
    api.post<Record<string, any>>('/v1/historical', { symbols, interval }),
  
  getFundamentals: (symbol: string) =>
    api.get<any>(`/v1/fundamentals/${symbol}`),
  
  getOrderBook: (symbol: string) =>
    api.get<OrderBookSnapshot>(`/v1/orderbook/${symbol}`),
  
  getTrades: (symbol: string, limit: number = 100) =>
    api.get<Trade[]>(`/v1/trades/${symbol}?limit=${limit}`),
}

// Simulation Engine API
export const simulationAPI = {
  placeOrder: (order: {
    symbol: string
    side: 'buy' | 'sell'
    order_type: 'market' | 'limit'
    quantity: number
    price?: number
    agent_id: string
  }) =>
    api.post<{ order: any; trades: Trade[]; status: string }>('/v1/orders', order),
  
  getOrder: (orderId: string) =>
    api.get<any>(`/v1/orders/${orderId}`),
  
  cancelOrder: (orderId: string) =>
    api.delete<{ status: string; order: any }>(`/v1/orders/${orderId}`),
}

// Agents API
export const agentsAPI = {
  getAgents: () =>
    api.get<Agent[]>('/v1/agents'),
  
  getAgent: (agentId: string) =>
    api.get<Agent>(`/v1/agents/${agentId}`),
  
  createAgent: (config: AgentConfig) =>
    api.post<{ status: string; agent_id: string }>('/v1/agents', config),
  
  startAgent: (agentId: string) =>
    api.post<{ status: string; agent_id: string }>(`/v1/agents/${agentId}/start`),
  
  stopAgent: (agentId: string) =>
    api.post<{ status: string; agent_id: string }>(`/v1/agents/${agentId}/stop`),
  
  deleteAgent: (agentId: string) =>
    api.delete<{ status: string; agent_id: string }>(`/v1/agents/${agentId}`),
}

// Risk Analytics API
export const riskAPI = {
  getRiskMetrics: (portfolioId: string) =>
    api.get<{ portfolio_id: string; metrics: Record<string, number>; timestamp: string }>(`/v1/risk/metrics/${portfolioId}`),
  
  calculateVar: (request: {
    portfolio_id: string
    metric_type: string
    confidence_level?: number
    time_horizon_days?: number
    lookback_days?: number
    simulations?: number
  }) =>
    api.post<{
      portfolio_id: string
      var_95: number
      var_99: number
      expected_shortfall_95: number
      expected_shortfall_99: number
      method: string
      timestamp: string
    }>('/v1/risk/var', request),
  
  stressTest: (portfolioId: string, scenario: string) =>
    api.get<{
      portfolio_id: string
      scenario: string
      portfolio_value_change: number
      stressed_var_95: number
      stressed_var_99: number
      stressed_expected_shortfall_95: number
      scenario_parameters: any
      timestamp: string
    }>(`/v1/risk/stress-test/${portfolioId}?scenario=${scenario}`),
  
  getCorrelationAnalysis: (portfolioId: string) =>
    api.get<{
      portfolio_id: string
      correlation_matrix: Record<string, Record<string, number>>
      symbols: string[]
      timestamp: string
    }>(`/v1/risk/correlation/${portfolioId}`),
  
  backtestVarModel: (portfolioId: string, confidenceLevel: number = 0.95, method: string = 'historical') =>
    api.post<{
      portfolio_id: string
      method: string
      confidence_level: number
      total_observations: number
      violations: number
      violation_rate: number
      expected_violation_rate: number
      kupiec_test_statistic: number
      model_accuracy: string
      timestamp: string
    }>('/v1/risk/backtesting', { portfolio_id: portfolioId, confidence_level: confidenceLevel, method }),
}

// Portfolio API
export const portfolioAPI = {
  optimizePortfolio: (request: {
    portfolio_id: string
    assets: Array<{
      symbol: string
      expected_return: number
      volatility: number
      current_price: number
      market_cap?: number
    }>
    method: string
    constraints: {
      min_weight?: number
      max_weight?: number
      max_concentration?: number
      min_positions?: number
      max_positions?: number
      target_risk?: number
      target_return?: number
    }
    risk_aversion?: number
    confidence_level?: number
  }) =>
    api.post<OptimizationResult>('/v1/portfolio/optimize', request),
  
  calculateEfficientFrontier: (request: any) =>
    api.post<{
      portfolio_id: string
      points: Array<{
        expected_return: number
        volatility: number
        sharpe_ratio: number
        weights: Record<string, number>
      }>
      optimal_portfolio: any
      timestamp: string
    }>('/v1/portfolio/efficient-frontier', request),
  
  suggestRebalancing: (portfolioId: string, frequency: string = 'monthly') =>
    api.get<{
      portfolio_id: string
      rebalancing_frequency: string
      suggestions: Array<{
        symbol: string
        current_weight: number
        target_weight: number
        action: string
        amount: number
      }>
      total_turnover: number
      estimated_cost: number
      timestamp: string
    }>(`/v1/portfolio/${portfolioId}/rebalance?frequency=${frequency}`),
  
  getRiskAttribution: (portfolioId: string) =>
    api.get<{
      portfolio_id: string
      risk_attribution: Record<string, number>
      total_risk: number
      systematic_risk: number
      idiosyncratic_risk: number
      timestamp: string
    }>(`/v1/portfolio/${portfolioId}/risk-attribution`),
}

export default api