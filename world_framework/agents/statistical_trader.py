

class StatisticalTrader(TradeAgent):
	def __init__(self, id, environment, lookback_period=10):
		super().__init__(id, environment)
		self.lookback_period = lookback_period