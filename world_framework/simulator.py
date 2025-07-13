
import time
import threading
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue 

from core.env import Environment
from core.economic_instrument import EconomicInstrument
from core.viz import RealTimeVisualizer
from core.iac import MessageBus

from agents.broker_agent import BrokerAgent
from agents.trader_agent import HeuristicTrader

class Simulator:
	def __init__(self, environment: Environment, instruments_config, enable_visualization=True):
		self.environment = environment
		self.enable_visualization = enable_visualization
		self.queue = Queue() if enable_visualization else None  # queue for communication between the processes
		
		if enable_visualization:
			self.visualizer = RealTimeVisualizer(self.queue, instruments_config)
			self.vis_interval = 1  # update every second
			self.last_visualization_time = time.time()
			self.visualizer_process = Process(target=self.visualizer.run)
		else:
			self.visualizer = None
			self.visualizer_process = None
			
		self.agents_threads = []

	def start_visualizer(self):
		if self.visualizer_process:
			self.visualizer_process.start()

	def update_visualizer(self):
		if not self.enable_visualization:
			return
			
		now = time.time()
		if now - self.last_visualization_time >= self.vis_interval:
			try:
				prices_data = {
					name: instrument.history[-100:]  # Send last 100 points
					for name, instrument in self.environment.instruments.items()
				}
				print(f"Simulator: Sending data to visualizer: {list(prices_data.keys())}")
				self.queue.put({"prices": prices_data}, timeout=1)  # Add timeout to prevent blocking
			except Exception as e:
				print(f"Simulator: Queue put failed: {e}")
			self.last_visualization_time = now

	def run(self, duration):
		# start agent threads
		for agent in self.environment.agents.values():
			agent.start()
			self.agents_threads.append(agent)

		if self.enable_visualization:
			self.start_visualizer()

		start_time = time.time()
		try:
			while time.time() - start_time < duration:
				# update instrument prices
				with self.environment._lock:
					for instrument in self.environment.instruments.values():
						instrument.update_prices()

				# publish market update
				self.environment.update_market()
				self.update_visualizer()
				time.sleep(self.environment.tick_duration)
		except KeyboardInterrupt:
			print("Simulation interrupted.")
			for agent in self.environment.agents.values():
				agent.stop()
		finally:
			print("Stopping simulation...")
			for agent in self.environment.agents.values():
				agent.stop()
			for thread in self.agents_threads:
				thread.join()

			if self.visualizer_process and self.visualizer_process.is_alive():
				self.visualizer.stop()
				self.visualizer_process.terminate()
				self.visualizer_process.join()
			if self.queue:
				self.queue.close()

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

	# Create simulator and run (disable visualization for headless mode)
	simulator = Simulator(env, instruments_config, enable_visualization=False)
	print("Simulation started. Press Ctrl+C to stop.")
	try:
		simulator.run(duration=10)  # Run for 10 seconds in test mode
		print("Simulation completed successfully!")
	except Exception as e:
		print(f"Simulation error: {e}")
		print("Note: Visualization may not work in headless environments")