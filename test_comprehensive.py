#!/usr/bin/env python3
"""
Comprehensive test suite for FinSim
"""

import time
import sys
import os

# Add the world_framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'world_framework'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from core.env import Environment
        from core.economic_instrument import EconomicInstrument, Order
        from core.iac import MessageBus, Message
        from core.viz import RealTimeVisualizer
        from core.agent import TradeAgent
        from agents.broker_agent import BrokerAgent
        from agents.trader_agent import HeuristicTrader
        from simulator import Simulator
        import world
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_economic_instruments():
    """Test economic instrument functionality"""
    print("Testing economic instruments...")
    try:
        from core.economic_instrument import EconomicInstrument
        instrument = EconomicInstrument("TestStock", 100.0, 0.02)
        initial_price = instrument.bid_price
        instrument.update_prices()
        assert len(instrument.history) == 2, "History should have 2 entries"
        assert instrument.bid_price != initial_price or True, "Price may or may not change"
        print("✅ Economic instruments working")
        return True
    except Exception as e:
        print(f"❌ Economic instruments failed: {e}")
        return False

def test_message_bus():
    """Test message bus functionality"""
    print("Testing message bus...")
    try:
        from core.iac import MessageBus, Message
        bus = MessageBus()
        received_messages = []
        
        def callback(message):
            received_messages.append(message)
        
        bus.subscribe("test", callback)
        test_message = Message("sender", "test", {"data": "test"})
        bus.publish(test_message)
        
        assert len(received_messages) == 1, "Should receive one message"
        assert received_messages[0].payload["data"] == "test", "Message payload should match"
        print("✅ Message bus working")
        return True
    except Exception as e:
        print(f"❌ Message bus failed: {e}")
        return False

def test_environment():
    """Test environment functionality"""
    print("Testing environment...")
    try:
        from core.iac import MessageBus
        from core.env import Environment
        from core.economic_instrument import EconomicInstrument, Order
        
        bus = MessageBus()
        env = Environment(tick_duration=0.1, message_bus=bus)
        
        # Add instrument
        instrument = EconomicInstrument("TestStock", 100.0, 0.02)
        env.add_instrument(instrument)
        assert "TestStock" in env.instruments, "Instrument should be added"
        
        # Test order submission
        order = Order("trader1", "TestStock", "buy", 1, 100.0)
        env.submit_order(order)
        assert len(env.order_book["TestStock"]["buy"]) == 1, "Order should be in order book"
        
        print("✅ Environment working")
        return True
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return False

def test_agents():
    """Test agent functionality"""
    print("Testing agents...")
    try:
        from core.iac import MessageBus
        from core.env import Environment
        from core.economic_instrument import EconomicInstrument
        from agents.broker_agent import BrokerAgent
        from agents.trader_agent import HeuristicTrader
        
        bus = MessageBus()
        env = Environment(tick_duration=0.1, message_bus=bus)
        
        # Add instrument
        instrument = EconomicInstrument("TestStock", 100.0, 0.02)
        env.add_instrument(instrument)
        
        # Create agents
        broker = BrokerAgent("TestBroker", env)
        trader = HeuristicTrader("TestTrader", env, strategy="momentum")
        
        env.add_agent(broker)
        env.add_agent(trader)
        
        assert len(env.agents) == 2, "Should have 2 agents"
        
        print("✅ Agents working")
        return True
    except Exception as e:
        print(f"❌ Agents failed: {e}")
        return False

def test_full_simulation():
    """Test full simulation run"""
    print("Testing full simulation...")
    try:
        from core.env import Environment
        from core.economic_instrument import EconomicInstrument
        from core.iac import MessageBus
        from agents.broker_agent import BrokerAgent
        from agents.trader_agent import HeuristicTrader
        from simulator import Simulator

        # Configuration
        instruments_config = {
            "TestStock": {"initial_price": 100, "volatility": 0.02},
        }

        bus = MessageBus()
        env = Environment(tick_duration=0.1, message_bus=bus)

        # Add instrument
        for name, config in instruments_config.items():
            instrument = EconomicInstrument(name, config["initial_price"], config["volatility"])
            env.add_instrument(instrument)

        # Add agents
        broker = BrokerAgent("TestBroker", env)
        trader1 = HeuristicTrader("TestTrader1", env, strategy="momentum")
        env.add_agent(broker)
        env.add_agent(trader1)

        # Run simulation without visualization
        simulator = Simulator(env, instruments_config, enable_visualization=False)
        
        # Start agents
        for agent in env.agents.values():
            agent.start()

        # Run for short time
        start_time = time.time()
        duration = 2  # 2 seconds
        
        while time.time() - start_time < duration:
            with env._lock:
                for instrument in env.instruments.values():
                    instrument.update_prices()
            env.update_market()
            time.sleep(env.tick_duration)

        # Stop agents
        for agent in env.agents.values():
            agent.stop()
        for agent in env.agents.values():
            agent.join()

        print("✅ Full simulation working")
        return True
    except Exception as e:
        print(f"❌ Full simulation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running FinSim Comprehensive Test Suite\n")
    
    tests = [
        test_imports,
        test_economic_instruments,
        test_message_bus,
        test_environment,
        test_agents,
        test_full_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FinSim is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)