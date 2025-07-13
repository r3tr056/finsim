import React, { useEffect } from 'react'
import {
  Box,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Text,
  VStack,
  HStack,
  Badge,
  Progress,
  Divider,
} from '@chakra-ui/react'
import { useDispatch, useSelector } from 'react-redux'
import { RootState, AppDispatch } from '../store/store'
import { fetchQuote } from '../store/marketSlice'
import { fetchAgents } from '../store/agentsSlice'

const Dashboard: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>()
  const { quotes } = useSelector((state: RootState) => state.market)
  const { agents } = useSelector((state: RootState) => state.agents)
  
  const watchSymbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

  useEffect(() => {
    // Fetch initial data
    watchSymbols.forEach(symbol => {
      dispatch(fetchQuote(symbol))
    })
    dispatch(fetchAgents())

    // Set up periodic updates
    const interval = setInterval(() => {
      watchSymbols.forEach(symbol => {
        dispatch(fetchQuote(symbol))
      })
    }, 5000)

    return () => clearInterval(interval)
  }, [dispatch])

  const runningAgents = agents.filter(agent => agent.status === 'running')
  const totalTrades = agents.reduce((sum, agent) => sum + agent.performance.totalTrades, 0)
  const totalPnL = agents.reduce((sum, agent) => sum + agent.performance.totalPnl, 0)

  return (
    <Box>
      <VStack spacing={6} align="stretch">
        <Box>
          <Heading size="lg" mb={2}>Dashboard</Heading>
          <Text color="gray.600">Real-time overview of your financial simulation</Text>
        </Box>

        {/* Market Overview */}
        <Card>
          <CardHeader>
            <Heading size="md">Market Overview</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, md: 2, lg: 5 }} spacing={4}>
              {watchSymbols.map(symbol => {
                const quote = quotes[symbol]
                if (!quote) return null
                
                const change = quote.change || 0
                const changePercent = quote.changePercent || 0
                
                return (
                  <Stat key={symbol}>
                    <StatLabel>{symbol}</StatLabel>
                    <StatNumber fontSize="xl">${quote.price.toFixed(2)}</StatNumber>
                    <StatHelpText>
                      <StatArrow type={change >= 0 ? 'increase' : 'decrease'} />
                      {changePercent.toFixed(2)}%
                    </StatHelpText>
                  </Stat>
                )
              })}
            </SimpleGrid>
          </CardBody>
        </Card>

        {/* System Overview */}
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Active Agents</StatLabel>
                <StatNumber>{runningAgents.length}</StatNumber>
                <StatHelpText>of {agents.length} total</StatHelpText>
              </Stat>
              <Progress 
                value={(runningAgents.length / Math.max(agents.length, 1)) * 100} 
                colorScheme="green" 
                size="sm" 
                mt={2} 
              />
            </CardBody>
          </Card>

          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Total Trades</StatLabel>
                <StatNumber>{totalTrades.toLocaleString()}</StatNumber>
                <StatHelpText>Today</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Total P&L</StatLabel>
                <StatNumber color={totalPnL >= 0 ? 'green.500' : 'red.500'}>
                  ${totalPnL.toLocaleString()}
                </StatNumber>
                <StatHelpText>
                  <StatArrow type={totalPnL >= 0 ? 'increase' : 'decrease'} />
                  Today
                </StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card>
            <CardBody>
              <Stat>
                <StatLabel>System Status</StatLabel>
                <StatNumber>
                  <Badge colorScheme="green" fontSize="md">Online</Badge>
                </StatNumber>
                <StatHelpText>All services running</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </SimpleGrid>

        {/* Agent Performance */}
        <Card>
          <CardHeader>
            <Heading size="md">Top Performing Agents</Heading>
          </CardHeader>
          <CardBody>
            <VStack spacing={4} align="stretch">
              {agents
                .filter(agent => agent.performance.totalTrades > 0)
                .sort((a, b) => b.performance.totalPnl - a.performance.totalPnl)
                .slice(0, 5)
                .map(agent => (
                  <Box key={agent.agentId}>
                    <HStack justify="space-between">
                      <VStack align="start" spacing={0}>
                        <Text fontWeight="medium">{agent.agentId}</Text>
                        <Text fontSize="sm" color="gray.600">
                          {agent.agentType} • {agent.strategy}
                        </Text>
                      </VStack>
                      <VStack align="end" spacing={0}>
                        <Text 
                          fontWeight="bold" 
                          color={agent.performance.totalPnl >= 0 ? 'green.500' : 'red.500'}
                        >
                          ${agent.performance.totalPnl.toLocaleString()}
                        </Text>
                        <Text fontSize="sm" color="gray.600">
                          {agent.performance.totalTrades} trades
                        </Text>
                      </VStack>
                    </HStack>
                    <Divider mt={2} />
                  </Box>
                ))}
              {agents.filter(agent => agent.performance.totalTrades > 0).length === 0 && (
                <Text color="gray.500" textAlign="center">
                  No trading activity yet
                </Text>
              )}
            </VStack>
          </CardBody>
        </Card>
      </VStack>
    </Box>
  )
}

export default Dashboard