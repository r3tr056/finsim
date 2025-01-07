
import time
import random
import threading

from queue import Queue
from collections import defaultdict
from core.economic_instrument import Order

class TradeAgent(threading.Thread):
	def __init__(self, id, environment, balance=10000, risk_tolerance=0.5):
		super().__init__()
		self.id = id
		self.environment = environment
		self.balance = balance
		self.portfolio = defaultdict(int)
		self.risk_tolerance = risk_tolerance
		self._lock = threading.Lock()
		self._stop_event = threading.Event()

		self.environment.message_bus.subscribe("trade_executed", self.on_trade_executed)
		self.environment.message_bus.subscribe("market_update", self.on_market_update)

	def place_order(self, instrument_name, order_type, quantity, price):
		instrument = self.environment.get_instrument(instrument_name)
		if not instrument:
			print(f"Agent {self.id}: Instrument {instrument_name} not found.")
			return

		order = Order(self.id, instrument_name, order_type, quantity, price)
		self.environment.submit_order(order)
		print(f"Agent {self.id}: Placed {order_type} order for {quantity} {instrument_name} at {price}")

	def update_portfolio(self, instrument_name, quantity_change, trade_price):
		with self._lock:
			self.portfolio[instrument_name] += quantity_change
			if quantity_change > 0:
				self.balance -= trade_price * quantity_change
			else:
				self.balance += trade_price * abs(quantity_change)

	def on_trade_executed(self, message):
		trade_data = message.payload
		if trade_data['buyer_id'] == self.id:
			self.update_portfolio(trade_data['instrument_name'], trade_data['quantity'], trade_data['price'])
		elif trade_data['seller_id'] == self.id:
			self.update_portfolio(trade_data['instrument_name'], -trade_data['quantity'], trade_data['price'])

	def on_market_update(self, message):
		pass

	def stop(self):
		self._stop_event.set()

	def stopped(self):
		return self._stop_event.is_set()

	def run(self):
		while not self.stopped():
			try:
				self.make_decision()
				time.sleep(random.uniform(0.1 * self.environment.tick_duration, self.environment.tick_duration)) # Simulate different reaction times
			except Exception as e:
				print(f"Agent {self.id} encountered an error: {e}")
				self.stop()

	def make_decision(self):
		raise NotImplementedError


