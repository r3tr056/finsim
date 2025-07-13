#!/usr/bin/env python3
"""
Simple test script to validate FinSim core functionality
"""

import time
import sys
import os

# Add the world_framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'world_framework'))

from core.env import Environment
from core.economic_instrument import EconomicInstrument
from core.iac import MessageBus
from agents.broker_agent import BrokerAgent
from agents.trader_agent import HeuristicTrader

def test_basic_simulation():
    """Test basic simulation functionality"""
    print("Testing basic FinSim functionality...")
    
    # Configuration for instruments
    instruments_config = {
        "TestStock": {"initial_price": 100, "volatility": 0.02},
        "TestBond": {"initial_price": 50, "volatility": 0.005},
    }

    message_bus = MessageBus()
    
    # Create environment
    env = Environment(tick_duration=0.1, message_bus=message_bus)
    
    # Add economic instruments
    for name, config in instruments_config.items():
        instrument = EconomicInstrument(name, config["initial_price"], config["volatility"])
        env.add_instrument(instrument)
        print(f"Added instrument: {name} (initial price: {config['initial_price']})")
    
    # Add agents
    broker = BrokerAgent("TestBroker", env)
    trader1 = HeuristicTrader("TestTrader1", env, strategy="momentum")
    trader2 = HeuristicTrader("TestTrader2", env, strategy="momentum")
    
    env.add_agent(broker)
    env.add_agent(trader1)
    env.add_agent(trader2)
    
    print(f"Added {len(env.agents)} agents")
    
    # Start agents
    for agent in env.agents.values():
        agent.start()
    
    # Run simulation for a short time
    start_time = time.time()
    duration = 5  # 5 seconds
    
    try:
        while time.time() - start_time < duration:
            # Update instrument prices
            with env._lock:
                for instrument in env.instruments.values():
                    instrument.update_prices()
            
            # Publish market update
            env.update_market()
            time.sleep(env.tick_duration)
    
    except KeyboardInterrupt:
        print("Test interrupted.")
    
    finally:
        print("Stopping agents...")
        for agent in env.agents.values():
            agent.stop()
        
        for agent in env.agents.values():
            agent.join()
        
        print("Test completed successfully!")
        
        # Print some statistics
        for name, instrument in env.instruments.items():
            print(f"{name}: Final price: {instrument.bid_price:.2f}, History length: {len(instrument.history)}")
        
        print(f"Total orders in system: {sum(len(orders['buy']) + len(orders['sell']) for orders in env.order_book.values())}")

if __name__ == "__main__":
    test_basic_simulation()