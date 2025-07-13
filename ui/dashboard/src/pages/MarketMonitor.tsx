import React from 'react'
import { Box, Heading, Text } from '@chakra-ui/react'

const MarketMonitor: React.FC = () => {
  return (
    <Box>
      <Heading size="lg" mb={4}>Market Monitor</Heading>
      <Text>Real-time market data and order book visualization</Text>
    </Box>
  )
}

export default MarketMonitor