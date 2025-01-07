'''
The Board represents the env where lifeforms can evolve, you can
initialize a Board by passing a tuple representing its size
'''

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from loguru import logger

from .lifeforms.base import LifeForm

# the channels
N_CH = 1
CHANNEL = range(N_CH)

class World:
	def __init__(self, size=(100, 100)):
		''' initialize the world '''
		self.names = ['', '', '']
		self.params = [{'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'h':1, 'r':1, 'kn':1, 'gn':1} for k in KERNEL]
		self.set_channels()
		self.param_P = 0
		self.size = size
		self.world_cells = [np.zeros(size, dtype=bool) for c in CHANNEL]

	def add(self, evolving_sub: Evolver, loc: Tuple[int, int]):
		''' add an evolving subject to the world '''
		try:
			h1, w1 = self.world_cells.shape
			h1, w2 = evolving_sub.cells.shape
			h, w = min(h1, h2), min(w1, w2)
			i1, j1 = (w1 - w) // 2 + shift[1], (h1 - h) // 2 + shift[0]
			i2, j2 = (w2 - w) // 2, (h2 - h) // 2
			vmin = np.amin(evolving_sub.cells)

			for y in range(h):
				for x in range(w):
					if evolving_sub.cells[j2 + y, i2 + x] > vmin:
						self.cells[(j1 + y) % h1, ]

			self.state[row: row + height, col: col + width] = evolving_sub.layout
		except ValueError:
			logger.error("Subject being added is out of world bounds")
			raise

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
			self.params = [WorldUtils.data_to_params(p) for p in params]
			self.set_channels()
		self.world_cells = None
		rle = data.get('world_cells')
		if rle is not None:
			if type(rle) not in [list]:
				rle = [rle for c in CHANNEL]
			self.world_cells = [WorldUtils.rle_2_cells(r) for r in rle]
		return self

	def clear(self):
		''' clear the world, reset everything '''
		logger.debug("World has be cleared and reset")
		self.world_cells.fill(0)

	def view(self, figsize=(5, 5)) -> Tuple[Figure, AxesImage]:
		''' view the current state of the world '''
		fig = plt.figure(figsize=figsize)
		ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
		im = ax.imshow(self.state, cmap=plt.cm.binary, interpolation="nearest")
		im.set_clim(-0.05, 1)
		return fig, im

	@staticmethod
	def rle_to_world_cells(st):
		stacks = 

