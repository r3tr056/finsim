import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from queue import Empty

class RealTimeVisualizer:
	def __init__(self, queue, instruments_config):
		self.queue = queue
		self.instruments_config = instruments_config
		self.running = True  # Flag to manage the process lifecycle

	def run(self):
		try:
			plt.ion()  # Enable interactive mode for real-time plotting
			fig, ax = plt.subplots()
			data = {name: [] for name in self.instruments_config.keys()}  # Initialize data

			while self.running:
				try:
					# Attempt to get data from the queue with a timeout
					incoming_data = self.queue.get(timeout=1)
					prices = incoming_data.get("prices", {})

					# Update data for plotting
					for name, price_history in prices.items():
						data[name] = price_history

					# Plot the updated data
					ax.clear()
					for name, price_history in data.items():
						ax.plot(price_history, label=name)
					ax.legend(loc="upper left")
					ax.set_title("Real-Time Market Visualization")
					ax.set_xlabel("Time")
					ax.set_ylabel("Price")
					plt.pause(0.01)  # Allow the plot to update
				except Empty:
					# Queue is empty; continue without crashing
					continue
				except Exception as e:
					print(f"Visualizer encountered an error: {e}")
					self.running = False

			plt.close(fig)
		except Exception as e:
			print(f"Visualization not available in headless mode: {e}")
			# Fall back to text-based reporting
			while self.running:
				try:
					incoming_data = self.queue.get(timeout=1)
					prices = incoming_data.get("prices", {})
					if prices:
						print("Market Update:")
						for name, price_history in prices.items():
							if price_history:
								print(f"  {name}: {price_history[-1]:.2f}")
				except Empty:
					continue
				except Exception as e:
					print(f"Visualizer error: {e}")
					self.running = False

	def stop(self):
		self.running = False
