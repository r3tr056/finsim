import { Routes, Route } from 'react-router-dom'
import { Box } from '@chakra-ui/react'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import MarketMonitor from './pages/MarketMonitor'
import AgentStudio from './pages/AgentStudio'
import RiskCenter from './pages/RiskCenter'
import PortfolioOptimizer from './pages/PortfolioOptimizer'
import Login from './pages/Login'
import { useSelector } from 'react-redux'
import { RootState } from './store/store'

function App() {
  const { isAuthenticated } = useSelector((state: RootState) => state.auth)

  if (!isAuthenticated) {
    return <Login />
  }

  return (
    <Box minH="100vh">
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/market" element={<MarketMonitor />} />
          <Route path="/agents" element={<AgentStudio />} />
          <Route path="/risk" element={<RiskCenter />} />
          <Route path="/portfolio" element={<PortfolioOptimizer />} />
        </Routes>
      </Layout>
    </Box>
  )
}

export default App