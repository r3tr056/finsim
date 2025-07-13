'''
The Board represents the env where lifeforms can evolve, you can
initialize a Board by passing a tuple representing its size
'''

from typing import Tuple
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from loguru import logger

# the channels
N_CH = 1
CHANNEL = range(N_CH)

# Default constants
DEF_R = 13
KERNEL = ['KERNEL1', 'KERNEL2', 'KERNEL3']

class World:
	def __init__(self, size=(100, 100)):
		''' initialize the world '''
		self.names = ['', '', '']
		self.params = [{'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'h':1, 'r':1, 'kn':1, 'gn':1} for k in KERNEL]
		self.set_channels()
		self.param_P = 0
		self.size = size
		self.world_cells = [np.zeros(size, dtype=bool) for c in CHANNEL]
		self.state = np.zeros(size, dtype=bool)  # Add state attribute

	def set_channels(self):
		''' set up channels for the world '''
		pass  # Implementation needed

	def add(self, evolving_sub, loc: Tuple[int, int]):
		''' add an evolving subject to the world '''
		try:
			# Simplified implementation to fix syntax errors
			row, col = loc
			if hasattr(evolving_sub, 'layout'):
				height, width = evolving_sub.layout.shape
				self.state[row: row + height, col: col + width] = evolving_sub.layout
		except (ValueError, AttributeError) as e:
			logger.error(f"Subject being added is out of world bounds: {e}")
			raise

	@classmethod
	def create_from_values(cls, world_cells, params=None, names=None):
		self = cls()
		self.names = names.copy() if names is not None else None
		self.params = copy.deepcopy(params) if params is not None else None
		self.world_cells = copy.deepcopy(world_cells) if world_cells is not None else None
		return self

	@classmethod
	def construct_world_from_data(cls, data):
		self = cls()
		self.names = [data.get('code', ''), data.get('name', ''), data.get('cname', '')]
		self.params = None
		params = data.get('params')
		if params is not None:
			if type(params) not in [list]:
				params = [params for K in KERNEL]
			self.params = [self.data_to_params(p) for p in params]
			self.set_channels()
		self.world_cells = None
		rle = data.get('world_cells')
		if rle is not None:
			if type(rle) not in [list]:
				rle = [rle for c in CHANNEL]
			self.world_cells = [self.rle_2_cells(r) for r in rle]
		return self

	def data_to_params(self, data):
		''' Convert data to parameters '''
		return data  # Simplified implementation

	def rle_2_cells(self, rle):
		''' Convert RLE to cells '''
		return np.zeros(self.size, dtype=bool)  # Simplified implementation

	def clear(self):
		''' clear the world, reset everything '''
		logger.debug("World has be cleared and reset")
		self.state.fill(0)
		for cells in self.world_cells:
			cells.fill(0)

	def view(self, figsize=(5, 5)) -> Tuple[Figure, AxesImage]:
		''' view the current state of the world '''
		fig = plt.figure(figsize=figsize)
		ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
		im = ax.imshow(self.state, cmap=plt.cm.binary, interpolation="nearest")
		im.set_clim(-0.05, 1)
		return fig, im

	@staticmethod
	def rle_to_world_cells(st):
		''' Convert RLE string to world cells '''
		# Simplified implementation - needs proper RLE decoding
		return np.zeros((100, 100), dtype=bool)

