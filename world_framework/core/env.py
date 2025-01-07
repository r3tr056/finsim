import threading
import random
import time

from collections import defaultdict
from core.iac import MessageBus, Message

class Environment:
	def __init__(self, tick_duration=1, message_bus=None):
		self.agents = {}
		self.instruments = {}
		self.order_book = defaultdict(lambda: {'buy': [], 'sell': []})
		self.message_bus = message_bus if message_bus else MessageBus()
		self.tick_duration = tick_duration
		self._lock = threading.Lock()

	def add_agent(self, agent):
		with self._lock:
			self.agents[agent.id] = agent

	def add_instrument(self, instrument):
		with self._lock:
			self.instruments[instrument.name] = instrument

	def get_instrument(self, name):
		with self._lock:
			return self.instruments.get(name)

	def submit_order(self, order):
		with self._lock:
			self.order_book[order.instrument_name][order.order_type].append(order)
			self.order_book[order.instrument_name][order.order_type].sort(
				key=lambda o: (o.price, -o.timestamp) if o.order_type == 'buy' else (o.price, o.timestamp)
			)
			self.message_bus.publish(Message(order.trader_id, "order_placed", order.__dict__))

	def update_market(self):
		with self._lock:
			market_data = {
				instrument_name: {'bid': instrument.bid_price, 'ask': instrument.ask_price} for instrument_name, instrument in self.instruments.items()
			}
			self.message_bus.publish(Message("environment", "market_update", market_data))