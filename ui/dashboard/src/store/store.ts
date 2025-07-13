import { configureStore } from '@reduxjs/toolkit'
import authSlice from './authSlice'
import marketSlice from './marketSlice'
import agentsSlice from './agentsSlice'
import portfolioSlice from './portfolioSlice'
import riskSlice from './riskSlice'
import uiSlice from './uiSlice'

export const store = configureStore({
  reducer: {
    auth: authSlice,
    market: marketSlice,
    agents: agentsSlice,
    portfolio: portfolioSlice,
    risk: riskSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch