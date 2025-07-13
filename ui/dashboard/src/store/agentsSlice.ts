import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { Agent, AgentConfig } from '../types'
import { agentsAPI } from '../utils/api'

interface AgentsState {
  agents: Agent[]
  loading: boolean
  error: string | null
}

const initialState: AgentsState = {
  agents: [],
  loading: false,
  error: null,
}

// Async thunks
export const fetchAgents = createAsyncThunk('agents/fetchAgents', async () => {
  const response = await agentsAPI.getAgents()
  return response.data
})

export const createAgent = createAsyncThunk(
  'agents/createAgent',
  async (config: AgentConfig) => {
    const response = await agentsAPI.createAgent(config)
    return response.data
  }
)

export const startAgent = createAsyncThunk(
  'agents/startAgent',
  async (agentId: string) => {
    const response = await agentsAPI.startAgent(agentId)
    return { agentId, status: 'running' }
  }
)

export const stopAgent = createAsyncThunk(
  'agents/stopAgent',
  async (agentId: string) => {
    const response = await agentsAPI.stopAgent(agentId)
    return { agentId, status: 'stopped' }
  }
)

export const deleteAgent = createAsyncThunk(
  'agents/deleteAgent',
  async (agentId: string) => {
    await agentsAPI.deleteAgent(agentId)
    return agentId
  }
)

const agentsSlice = createSlice({
  name: 'agents',
  initialState,
  reducers: {
    updateAgentPerformance: (state, action: PayloadAction<{ agentId: string; performance: Agent['performance'] }>) => {
      const agent = state.agents.find(a => a.agentId === action.payload.agentId)
      if (agent) {
        agent.performance = action.payload.performance
      }
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAgents.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchAgents.fulfilled, (state, action) => {
        state.loading = false
        state.agents = action.payload
      })
      .addCase(fetchAgents.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Failed to fetch agents'
      })
      .addCase(createAgent.fulfilled, (state, action) => {
        // Add the new agent to the list
        const newAgent: Agent = {
          agentId: action.meta.arg.agentId,
          agentType: action.meta.arg.agentType,
          strategy: action.meta.arg.strategy,
          symbols: action.meta.arg.symbols,
          enabled: action.meta.arg.enabled,
          status: 'stopped',
          performance: {
            totalPnl: 0,
            totalTrades: 0,
            winRate: 0,
            sharpeRatio: 0,
          },
          timestamp: new Date().toISOString(),
        }
        state.agents.push(newAgent)
      })
      .addCase(startAgent.fulfilled, (state, action) => {
        const agent = state.agents.find(a => a.agentId === action.payload.agentId)
        if (agent) {
          agent.status = 'running'
        }
      })
      .addCase(stopAgent.fulfilled, (state, action) => {
        const agent = state.agents.find(a => a.agentId === action.payload.agentId)
        if (agent) {
          agent.status = 'stopped'
        }
      })
      .addCase(deleteAgent.fulfilled, (state, action) => {
        state.agents = state.agents.filter(a => a.agentId !== action.payload)
      })
  },
})

export const { updateAgentPerformance, clearError } = agentsSlice.actions
export default agentsSlice.reducer