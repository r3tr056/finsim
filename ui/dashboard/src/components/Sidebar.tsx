import React from 'react'
import {
  Box,
  VStack,
  HStack,
  Text,
  Icon,
  useColorModeValue,
  Flex,
} from '@chakra-ui/react'
import { Link, useLocation } from 'react-router-dom'
import {
  FiHome,
  FiTrendingUp,
  FiUsers,
  FiShield,
  FiPieChart,
  FiActivity,
} from 'react-icons/fi'

interface SidebarProps {
  isOpen: boolean
}

const navItems = [
  { name: 'Dashboard', icon: FiHome, path: '/dashboard' },
  { name: 'Market Monitor', icon: FiTrendingUp, path: '/market' },
  { name: 'Agent Studio', icon: FiUsers, path: '/agents' },
  { name: 'Risk Center', icon: FiShield, path: '/risk' },
  { name: 'Portfolio Optimizer', icon: FiPieChart, path: '/portfolio' },
  { name: 'Analytics', icon: FiActivity, path: '/analytics' },
]

const Sidebar: React.FC<SidebarProps> = ({ isOpen }) => {
  const location = useLocation()
  const bg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  return (
    <Box
      bg={bg}
      borderRight="1px"
      borderColor={borderColor}
      w={isOpen ? '240px' : '60px'}
      h="calc(100vh - 64px)"
      position="sticky"
      top="16"
      transition="width 0.2s"
      overflow="hidden"
    >
      <VStack spacing={1} align="stretch" p={2}>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path
          return (
            <Link key={item.name} to={item.path}>
              <Flex
                align="center"
                p={3}
                mx={2}
                borderRadius="md"
                role="group"
                cursor="pointer"
                bg={isActive ? 'brand.500' : 'transparent'}
                color={isActive ? 'white' : undefined}
                _hover={{
                  bg: isActive ? 'brand.600' : 'gray.100',
                  color: isActive ? 'white' : 'brand.500',
                }}
                transition="all 0.2s"
              >
                <Icon
                  mr={isOpen ? 4 : 0}
                  fontSize="16"
                  as={item.icon}
                />
                {isOpen && (
                  <Text fontSize="sm" fontWeight="medium">
                    {item.name}
                  </Text>
                )}
              </Flex>
            </Link>
          )
        })}
      </VStack>
    </Box>
  )
}

export default Sidebar