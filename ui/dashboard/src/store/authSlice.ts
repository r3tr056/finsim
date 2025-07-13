import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { AuthState, User, ApiResponse } from '../types'
import { authAPI } from '../utils/api'

const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  permissions: [],
}

// Async thunks
export const login = createAsyncThunk(
  'auth/login',
  async (credentials: { email: string; password: string }) => {
    const response = await authAPI.login(credentials)
    localStorage.setItem('token', response.data.access_token)
    return response.data
  }
)

export const logout = createAsyncThunk('auth/logout', async () => {
  localStorage.removeItem('token')
  return null
})

export const getCurrentUser = createAsyncThunk('auth/getCurrentUser', async () => {
  const response = await authAPI.getCurrentUser()
  return response.data
})

export const getUserPermissions = createAsyncThunk('auth/getUserPermissions', async () => {
  const response = await authAPI.getUserPermissions()
  return response.data.permissions
})

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setToken: (state, action: PayloadAction<string>) => {
      state.token = action.payload
      localStorage.setItem('token', action.payload)
    },
    clearAuth: (state) => {
      state.isAuthenticated = false
      state.user = null
      state.token = null
      state.permissions = []
      localStorage.removeItem('token')
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(login.fulfilled, (state, action) => {
        state.isAuthenticated = true
        state.token = action.payload.access_token
      })
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
        state.permissions = []
      })
      .addCase(getCurrentUser.fulfilled, (state, action) => {
        state.user = action.payload
        state.isAuthenticated = true
      })
      .addCase(getUserPermissions.fulfilled, (state, action) => {
        state.permissions = action.payload
      })
  },
})

export const { setToken, clearAuth } = authSlice.actions
export default authSlice.reducer