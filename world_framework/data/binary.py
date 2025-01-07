# data types
binary = Binary()

class Atomic(metaclass=abc.ABCMeta):
	''' abc objects that can be RLP encoded '''
	pass

Atomic.register(bytes)
Atomic.register(bytearray)

class Binary(object):
	''' A sedes object for binary data for the world of certain length '''
	def __init__(self, min_len=None, max_len=None, allow_empty=False):
		self.min_len = min_len or 0
		if max_len is None:
			self.max_len = float('inf')
		else:
			self.max_len = max_len
		self.allow_empty = allow_empty

	@classmethod
	def fixed_length(cls, l, allow_empty=False):
		return cls(l, l, allow_empty=allow_empty)

	@classmethod
	def is_valid_type(cls, obj):
		return isinstance(obj, (bytes, bytearray))

	def is_valid_length(self, l):
		return any((self.min_len <= l <= self.max_len, self.allow_empty and l == 0))

	def serialize(self, obj):
		if not Binary.is_valid_type(obj):
			raise SerializationError('Object is not serializable ({})'.format(type(obj)), obj)
		if not self.is_valid_length(len(obj)):
			raise SerializationError('Object has invalid length', obj)
		return obj

	def deserialize(self, serial):
		if not isinstance(serial, Atomic):
			m = 'Objects of type {} cannot be deserialized'
			raise DeserializationError(m.format(type(serial).__name__), serial)
		if self.is_valid_length(len(serial)):
			return serial
		else:
			raise DeserializationError('{} has invalid length'.format(type(serial)), serial)

