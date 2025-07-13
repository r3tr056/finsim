import React, { useState } from 'react'
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Text,
  Alert,
  AlertIcon,
  useToast,
  Flex,
  Heading,
  Card,
  CardBody,
} from '@chakra-ui/react'
import { useDispatch } from 'react-redux'
import { login } from '../store/authSlice'
import { AppDispatch } from '../store/store'

const Login: React.FC = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  const dispatch = useDispatch<AppDispatch>()
  const toast = useToast()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      await dispatch(login({ email, password })).unwrap()
      toast({
        title: 'Login successful',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Flex minH="100vh" align="center" justify="center" bg="gray.50" _dark={{ bg: 'gray.900' }}>
      <Box w="full" maxW="md" p={6}>
        <Card>
          <CardBody>
            <VStack spacing={6}>
              <Heading size="lg" textAlign="center" color="brand.500">
                FinSim Dashboard
              </Heading>
              
              <Text color="gray.600" textAlign="center">
                Sign in to access your financial simulation platform
              </Text>

              {error && (
                <Alert status="error" borderRadius="md">
                  <AlertIcon />
                  {error}
                </Alert>
              )}

              <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                <VStack spacing={4}>
                  <FormControl isRequired>
                    <FormLabel>Email</FormLabel>
                    <Input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="Enter your email"
                    />
                  </FormControl>

                  <FormControl isRequired>
                    <FormLabel>Password</FormLabel>
                    <Input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Enter your password"
                    />
                  </FormControl>

                  <Button
                    type="submit"
                    colorScheme="brand"
                    size="lg"
                    width="full"
                    isLoading={loading}
                    loadingText="Signing in..."
                  >
                    Sign In
                  </Button>
                </VStack>
              </form>

              <Box textAlign="center" pt={4}>
                <Text fontSize="sm" color="gray.600">
                  Demo credentials:
                </Text>
                <Text fontSize="sm" fontFamily="mono">
                  admin@finsim.com / admin123
                </Text>
              </Box>
            </VStack>
          </CardBody>
        </Card>
      </Box>
    </Flex>
  )
}

export default Login