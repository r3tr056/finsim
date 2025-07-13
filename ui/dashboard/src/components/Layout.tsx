import React from 'react'
import { Box, Flex } from '@chakra-ui/react'
import { useSelector, useDispatch } from 'react-redux'
import { RootState } from '../store/store'
import { toggleSidebar } from '../store/uiSlice'
import Header from './Header'
import Sidebar from './Sidebar'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const dispatch = useDispatch()
  const { sidebarOpen } = useSelector((state: RootState) => state.ui)

  const handleSidebarToggle = () => {
    dispatch(toggleSidebar())
  }

  return (
    <Box minH="100vh">
      <Header onSidebarToggle={handleSidebarToggle} />
      <Flex>
        <Sidebar isOpen={sidebarOpen} />
        <Box flex="1" p={6} overflow="auto">
          {children}
        </Box>
      </Flex>
    </Box>
  )
}

export default Layout