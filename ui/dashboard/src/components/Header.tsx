import React from 'react'
import {
  Box,
  Flex,
  HStack,
  IconButton,
  Text,
  useColorModeValue,
  Avatar,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider,
  useColorMode,
  Spacer,
  Badge,
} from '@chakra-ui/react'
import { HamburgerIcon, SunIcon, MoonIcon, BellIcon } from '@chakra-ui/icons'
import { useDispatch, useSelector } from 'react-redux'
import { RootState } from '../store/store'
import { toggleSidebar, toggleColorMode } from '../store/uiSlice'
import { logout } from '../store/authSlice'

interface HeaderProps {
  onSidebarToggle: () => void
}

const Header: React.FC<HeaderProps> = ({ onSidebarToggle }) => {
  const { colorMode, toggleColorMode: toggleChakraColorMode } = useColorMode()
  const dispatch = useDispatch()
  const { user } = useSelector((state: RootState) => state.auth)
  const { notifications } = useSelector((state: RootState) => state.ui)
  
  const bg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  const handleLogout = () => {
    dispatch(logout())
  }

  const handleColorModeToggle = () => {
    toggleChakraColorMode()
    dispatch(toggleColorMode())
  }

  const unreadNotifications = notifications.filter(n => n.type === 'error' || n.type === 'warning').length

  return (
    <Box
      bg={bg}
      borderBottom="1px"
      borderColor={borderColor}
      px={4}
      height="16"
      position="sticky"
      top={0}
      zIndex={10}
    >
      <Flex h="16" alignItems="center" justifyContent="space-between">
        <HStack spacing={4}>
          <IconButton
            size="md"
            icon={<HamburgerIcon />}
            aria-label="Toggle sidebar"
            variant="ghost"
            onClick={onSidebarToggle}
          />
          <Text fontSize="xl" fontWeight="bold" color="brand.500">
            FinSim
          </Text>
        </HStack>

        <Spacer />

        <HStack spacing={4}>
          <IconButton
            size="md"
            icon={colorMode === 'light' ? <MoonIcon /> : <SunIcon />}
            aria-label="Toggle color mode"
            variant="ghost"
            onClick={handleColorModeToggle}
          />
          
          <Box position="relative">
            <IconButton
              size="md"
              icon={<BellIcon />}
              aria-label="Notifications"
              variant="ghost"
            />
            {unreadNotifications > 0 && (
              <Badge
                colorScheme="red"
                position="absolute"
                top="-1"
                right="-1"
                fontSize="xs"
                borderRadius="full"
              >
                {unreadNotifications}
              </Badge>
            )}
          </Box>

          <Menu>
            <MenuButton>
              <Avatar
                size="sm"
                name={user?.fullName || 'User'}
                src=""
                cursor="pointer"
              />
            </MenuButton>
            <MenuList>
              <MenuItem>
                <Box>
                  <Text fontWeight="medium">{user?.fullName}</Text>
                  <Text fontSize="sm" color="gray.500">{user?.email}</Text>
                </Box>
              </MenuItem>
              <MenuDivider />
              <MenuItem>Profile</MenuItem>
              <MenuItem>Settings</MenuItem>
              <MenuDivider />
              <MenuItem onClick={handleLogout}>Sign out</MenuItem>
            </MenuList>
          </Menu>
        </HStack>
      </Flex>
    </Box>
  )
}

export default Header