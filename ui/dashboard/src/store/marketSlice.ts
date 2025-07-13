import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { QuoteData, OrderBookSnapshot, Trade } from '../types'
import { marketAPI } from '../utils/api'

interface MarketState {
  quotes: Record<string, QuoteData>
  orderBooks: Record<string, OrderBookSnapshot>
  trades: Trade[]
  isConnected: boolean
  loading: boolean
  error: string | null
}

const initialState: MarketState = {
  quotes: {},
  orderBooks: {},
  trades: [],
  isConnected: false,
  loading: false,
  error: null,
}

// Async thunks
export const fetchQuote = createAsyncThunk(
  'market/fetchQuote',
  async (symbol: string) => {
    const response = await marketAPI.getQuote(symbol)
    return response.data
  }
)

export const fetchOrderBook = createAsyncThunk(
  'market/fetchOrderBook',
  async (symbol: string) => {
    const response = await marketAPI.getOrderBook(symbol)
    return response.data
  }
)

export const fetchTrades = createAsyncThunk(
  'market/fetchTrades',
  async (symbol: string) => {
    const response = await marketAPI.getTrades(symbol)
    return response.data
  }
)

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    updateQuote: (state, action: PayloadAction<QuoteData>) => {
      state.quotes[action.payload.symbol] = action.payload
    },
    updateOrderBook: (state, action: PayloadAction<OrderBookSnapshot>) => {
      state.orderBooks[action.payload.symbol] = action.payload
    },
    addTrade: (state, action: PayloadAction<Trade>) => {
      state.trades.unshift(action.payload)
      // Keep only last 1000 trades
      if (state.trades.length > 1000) {
        state.trades = state.trades.slice(0, 1000)
      }
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchQuote.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchQuote.fulfilled, (state, action) => {
        state.loading = false
        state.quotes[action.payload.symbol] = action.payload
      })
      .addCase(fetchQuote.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Failed to fetch quote'
      })
      .addCase(fetchOrderBook.fulfilled, (state, action) => {
        state.orderBooks[action.payload.symbol] = action.payload
      })
      .addCase(fetchTrades.fulfilled, (state, action) => {
        state.trades = action.payload
      })
  },
})

export const { updateQuote, updateOrderBook, addTrade, setConnectionStatus, clearError } = marketSlice.actions
export default marketSlice.reducer