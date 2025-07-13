#!/usr/bin/env python3
"""
Test the simulator without visualization for headless environments
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

class HeadlessSimulator:
    def __init__(self, environment: Environment, instruments_config):
        self.environment = environment
        self.agents_threads = []

    def run(self, duration):
        # start agent threads
        for agent in self.environment.agents.values():
            agent.start()
            self.agents_threads.append(agent)

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                # update instrument prices
                with self.environment._lock:
                    for instrument in self.environment.instruments.values():
                        instrument.update_prices()

                # publish market update
                self.environment.update_market()
                time.sleep(self.environment.tick_duration)
        except KeyboardInterrupt:
            print("Simulation interrupted.")
        finally:
            print("Stopping simulation...")
            for agent in self.environment.agents.values():
                agent.stop()
            for thread in self.agents_threads:
                thread.join()

# Main simulation setup
if __name__ == "__main__":
    # Configuration for instruments
    instruments_config = {
        "StockA": {"initial_price": 100, "volatility": 0.02},
        "StockB": {"initial_price": 200, "volatility": 0.015},
        "BondX": {"initial_price": 50, "volatility": 0.005},
    }

    message_bus = MessageBus()

    # Create environment
    env = Environment(tick_duration=0.1, message_bus=message_bus)

    # Add economic instruments
    for name, config in instruments_config.items():
        instrument = EconomicInstrument(name, config["initial_price"], config["volatility"])
        env.add_instrument(instrument)

    # Add agents
    broker = BrokerAgent("Broker1", env)
    trader1 = HeuristicTrader("Trader1", env, strategy="momentum")
    trader2 = HeuristicTrader("Trader2", env, strategy="momentum")
    env.add_agent(broker)
    env.add_agent(trader1)
    env.add_agent(trader2)

    # Create simulator and run
    simulator = HeadlessSimulator(env, instruments_config)
    print("Headless simulation started. Running for 10 seconds...")
    simulator.run(duration=10)
    print("Simulation completed successfully!")