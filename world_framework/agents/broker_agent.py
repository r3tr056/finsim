
import time
import random

from core.agent import TradeAgent
from core.iac import Message

class BrokerAgent(TradeAgent):
	def __init__(self, id, environment, initial_balance=10000, commission_rate=0.01):
		super().__init__(id, environment, balance=initial_balance)
		self.commission_rate = commission_rate
		self.environment.message_bus.subscribe("order_placed", self.on_order_placed)

	def match_orders(self):
		with self.environment._lock:
			for instrument_name, orders in self.environment.order_book.items():
				buy_orders = sorted(orders['buy'], key=lambda o: o.price, reverse=True)
				sell_orders = sorted(orders['sell'], key=lambda o: o.price)

				buy_idx, sell_idx = 0, 0
				while buy_idx < len(buy_orders) and sell_idx < len(sell_orders) and buy_orders[buy_idx].price >= sell_orders[sell_idx].price:
					buy_order = buy_orders[buy_idx]
					sell_order = sell_orders[sell_idx]
					trade_price = sell_order.price  # Using sell order price for simplicity (can be more complex)
					quantity = min(buy_order.quantity, sell_order.quantity)

					buyer = self.environment.agents.get(buy_order.trader_id)
					seller = self.environment.agents.get(sell_order.trader_id)

					if buyer and seller:
						commission_buyer = trade_price * quantity * self.commission_rate
						commission_seller = trade_price * quantity * self.commission_rate

						with buyer._lock:
							can_buy = buyer.balance >= trade_price * quantity + commission_buyer
						with seller._lock:
							can_sell = seller.portfolio.get(instrument_name, 0) >= quantity

						# Ensure agents have enough funds (simplified check)
						if can_buy and can_sell:
							buyer.balance -= trade_price * quantity + commission_buyer
							seller.balance += trade_price * quantity - commission_seller
							buyer.portfolio[instrument_name] = buyer.portfolio.get(instrument_name, 0) + quantity
							seller.portfolio[instrument_name] -= quantity

							print(f"Broker {self.id}: Trade executed - {quantity} {instrument_name} at {trade_price}. Buyer: {buyer.id}, Seller: {seller.id}")
							self.environment.message_bus.publish(Message(self.id, "trade_executed", {
								"instrument_name": instrument_name,
								"price": trade_price,
								"quantity": quantity,
								"buyer_id": buy_order.trader_id,
								"seller_id": sell_order.trader_id
							}))

							# Update order quantities
							buy_order.quantity -= quantity
							sell_order.quantity -= quantity

					if buy_order.quantity == 0:
						buy_idx += 1
					if sell_order.quantity == 0:
						sell_idx += 1

				# Remove fully executed orders (important for maintaining order book)
				self.environment.order_book[instrument_name]['buy'] = [order for order in self.environment.order_book[instrument_name]['buy'] if order.quantity > 0]
				self.environment.order_book[instrument_name]['sell'] = [order for order in self.environment.order_book[instrument_name]['sell'] if order.quantity > 0]

	def on_order_placed(self, message):
		pass

	def run(self):
		while not self.stopped():
			try:
				self.match_orders()
				time.sleep(0.5 * self.environment.tick_duration) # Broker acts more frequently
			except Exception as e:
				print(f"Broker {self.id} encountered an error: {e}")
				self.stop()