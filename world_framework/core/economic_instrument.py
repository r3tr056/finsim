import json
import time
import random
import threading

class EconomicInstrument:
	def __init__(self, name, initial_price, volatility=0.01):
		self.name = name
		self.bid_price = initial_price  # Best bid price
		self.ask_price = initial_price * 1.005  # Best ask price (slightly higher)
		self.history = [(0, initial_price)]  # (timestamp, price)
		self.volatility = volatility
		self._lock = threading.Lock()

	def update_prices(self):
		with self._lock:
			# simplified volatility model
			price_change = random.uniform(-self.volatility, self.volatility) * self.ask_price
			self.bid_price += price_change
			self.ask_price = self.bid_price * (1 + random.uniform(0.003, 0.007))  # Keep ask slightly above bid
			self.history.append((time.time(), self.bid_price))

	def get_current_bid_ask(self):
		with self._lock:
			return self.bid_price, self.ask_price

class Order:
	def __init__(self, trader_id, instrument_name, order_type, quantity, price, order_time=None):
		self.trader_id = trader_id
		self.instrument_name = instrument_name
		self.order_type = order_type
		self.quantity = quantity
		self.price = price
		self.timestamp = order_time if order_time else time.time()
