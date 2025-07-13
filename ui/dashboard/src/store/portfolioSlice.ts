import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { Portfolio, OptimizationResult } from '../types'
import { portfolioAPI } from '../utils/api'

interface PortfolioState {
  portfolios: Portfolio[]
  optimizationResults: OptimizationResult[]
  loading: boolean
  error: string | null
}

const initialState: PortfolioState = {
  portfolios: [],
  optimizationResults: [],
  loading: false,
  error: null,
}

export const fetchPortfolios = createAsyncThunk('portfolio/fetchPortfolios', async () => {
  // In a real app, this would fetch from API
  return []
})

export const optimizePortfolio = createAsyncThunk(
  'portfolio/optimize',
  async (request: any) => {
    const response = await portfolioAPI.optimizePortfolio(request)
    return response.data
  }
)

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(optimizePortfolio.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(optimizePortfolio.fulfilled, (state, action) => {
        state.loading = false
        state.optimizationResults.unshift(action.payload)
      })
      .addCase(optimizePortfolio.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Portfolio optimization failed'
      })
  },
})

export const { clearError } = portfolioSlice.actions
export default portfolioSlice.reducer