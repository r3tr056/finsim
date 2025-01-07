import time
import threading
from collections import defaultdict

class Message:
	def __init__(self, sender_id, topic, payload):
		self.sender_id = sender_id
		self.topic = topic
		self.payload = payload
		self.timestamp = time.time()

class MessageBus:
	def __init__(self):
		self.subscriptions = defaultdict(list)
		self._lock = threading.Lock()

	def subscribe(self, topic, callback):
		with self._lock:
			self.subscriptions[topic].append(callback)

	def publish(self, message):
		with self._lock:
			if message.topic in self.subscriptions:
				for callback in self.subscriptions[message.topic]:
					callback(message)

