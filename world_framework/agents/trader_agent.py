import time
import random
from core.agent import TradeAgent

from collections import defaultdict


class HeuristicTrader(TradeAgent):
	def __init__(self, id, environment, strategy="momentum"):
		super().__init__(id, environment)
		self.strategy = strategy

	def make_decision(self):
		for instrument_name, instrument in self.environment.instruments.items():
			bid, ask = instrument.get_current_bid_ask()
			history = [h[1] for h in instrument.history]
			if not history or len(history) < 2:
				continue
			previous_price = history[-2]

			if self.strategy == "momentum":
				if bid > previous_price:
					self.place_order(instrument_name, 'buy', 1, ask * (1 + random.uniform(0.0001, 0.0005))) # Buy slightly above ask
				elif ask < previous_price:
					self.place_order(instrument_name, 'sell', 1, bid * (1 - random.uniform(0.0001, 0.0005))) # Sell slightly below bid
			elif self.strategy == "contrarian":
				if bid < previous_price:
					self.place_order(instrument_name, 'buy', 1, ask * (1 + random.uniform(0.0001, 0.0005)))
				elif ask > previous_price:
					self.place_order(instrument_name, 'sell', 1, bid * (1 - random.uniform(0.0001, 0.0005)))
