import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { RiskMetrics } from '../types'
import { riskAPI } from '../utils/api'

interface RiskState {
  riskMetrics: Record<string, RiskMetrics>
  loading: boolean
  error: string | null
}

const initialState: RiskState = {
  riskMetrics: {},
  loading: false,
  error: null,
}

export const fetchRiskMetrics = createAsyncThunk(
  'risk/fetchRiskMetrics',
  async (portfolioId: string) => {
    const response = await riskAPI.getRiskMetrics(portfolioId)
    return { portfolioId, metrics: response.data }
  }
)

export const calculateVar = createAsyncThunk(
  'risk/calculateVar',
  async (request: { portfolioId: string; confidenceLevel: number; method: string }) => {
    const response = await riskAPI.calculateVar(request)
    return response.data
  }
)

const riskSlice = createSlice({
  name: 'risk',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchRiskMetrics.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchRiskMetrics.fulfilled, (state, action) => {
        state.loading = false
        state.riskMetrics[action.payload.portfolioId] = action.payload.metrics
      })
      .addCase(fetchRiskMetrics.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Failed to fetch risk metrics'
      })
  },
})

export const { clearError } = riskSlice.actions
export default riskSlice.reducer