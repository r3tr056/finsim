// Market Data Types
export interface QuoteData {
  symbol: string
  price: number
  bid: number
  ask: number
  volume: number
  timestamp: string
  exchange: string
  change?: number
  changePercent?: number
}

export interface OrderBookLevel {
  price: number
  quantity: number
}

export interface OrderBookSnapshot {
  symbol: string
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  lastTradePrice?: number
  timestamp: string
}

// Trading Types
export interface Order {
  orderId: string
  symbol: string
  side: 'buy' | 'sell'
  orderType: 'market' | 'limit'
  quantity: number
  price?: number
  timestamp: string
  agentId: string
  status: 'pending' | 'filled' | 'partial' | 'cancelled'
  filledQuantity: number
}

export interface Trade {
  tradeId: string
  symbol: string
  price: number
  quantity: number
  buyOrderId: string
  sellOrderId: string
  timestamp: string
}

// Agent Types
export interface Agent {
  agentId: string
  agentType: 'heuristic' | 'lstm' | 'transformer' | 'gru' | 'dqn' | 'ppo' | 'a3c'
  strategy?: 'momentum' | 'mean_reversion' | 'value' | 'arbitrage'
  symbols: string[]
  enabled: boolean
  status: 'running' | 'stopped'
  performance: {
    totalPnl: number
    totalTrades: number
    winRate: number
    sharpeRatio: number
  }
  lastAction?: string
  timestamp: string
}

export interface AgentConfig {
  agentId: string
  agentType: Agent['agentType']
  strategy?: Agent['strategy']
  symbols: string[]
  parameters: Record<string, any>
  enabled: boolean
}

// Portfolio Types
export interface PortfolioPosition {
  symbol: string
  quantity: number
  currentPrice: number
  marketValue: number
  weight: number
}

export interface Portfolio {
  portfolioId: string
  positions: PortfolioPosition[]
  totalValue: number
  timestamp: string
}

export interface OptimizationResult {
  portfolioId: string
  method: 'mean_variance' | 'black_litterman' | 'risk_parity' | 'minimum_variance' | 'maximum_sharpe'
  allocations: PortfolioPosition[]
  expectedReturn: number
  expectedVolatility: number
  sharpeRatio: number
  totalValue: number
  timestamp: string
}

// Risk Types
export interface RiskMetrics {
  portfolioId: string
  var95: number
  var99: number
  expectedShortfall95: number
  expectedShortfall99: number
  volatility: number
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  maxDrawdown: number
  timestamp: string
}

// Auth Types
export interface User {
  id: string
  email: string
  fullName: string
  role: 'admin' | 'trader' | 'analyst' | 'viewer'
  isActive: boolean
  createdAt: string
  lastLogin?: string
}

export interface AuthState {
  isAuthenticated: boolean
  user: User | null
  token: string | null
  permissions: string[]
}

// WebSocket Types
export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

// API Response Types
export interface ApiResponse<T = any> {
  data: T
  message?: string
  status: 'success' | 'error'
  timestamp: string
}

export interface ApiError {
  message: string
  code?: string
  details?: any
}