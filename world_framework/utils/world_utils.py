import matplotlib.pyplot as plt
import numpy as np

WORLD_MATH_KERNELS = {}

def plot(ca, title=""):
	cmap = plt.get_cmap('Greys')
	plt.title(title)
	plt.imshow(ca, interpolation='none', cmap=cmap)
	plt.show()

def plot_multiple(ca_list, titles):
	cmap = plt.get_cmap('Greys')
	for i in range(0, len(ca_list)):
		plt.figure(i)
		plt.title(titles[i])
		plt.imshow(ca_list[i], interpolation='none', cmap=cmap)

def from_data(data: str):
	world = World()
	world.names = [data.get('code', ''), data.get('name', ''), data.get('cname', '')]
	world.params = None
	params = data.get('params')
	if params is not None:
		if type(params) not in [list]:
			params = [params for k in WORLD_MATH_KERNELS]
		world.params = [World.data_to_params(p) for p in params]
		world.set_channels()
	# empty the world cells
	world.world_cells = None
	rlpx_cells = data.get('world_cells')
	if rlpx_cells is not None:
		if type(rlpx_cells) not in [list]:
			rlpx_cells = [rlpx_cells for c in CHANNEL]
		world.world_cells = [World.rlpx_to_cells(r) for r in rlpx_cells]
	return world

def world_to_data(world, is_shorten=True):
	world_cell_rlpx = [World.world_cells_to_rlpx(world.world_cells[c], is_shorten) for C in world.CHANNELS]
	world_params = [World.params_to_data(world.params[k]) for k in world.MATH_KERNELS]
	data = {'code': world.names[0], 'name': world.names[1], 'cname': self.names[2], 'params': params, 'world_cells': world_rlpx}
	# compile the data, and compress it
	return data

def world_cells_to_rlpx(world_cells, is_shorten=True):
	values = np.rint(world_cells * 255).astype(int).tolist()
	if is_shorten:
		rlpx_grps = World._recur_drill_list(0, values, lambda: row: [(len(list(g)), World.val_to_ch(v).strip()) for f, g in itertools.groupby(row)])
		st = World._recur_join_st(0, )

def to_data(world: World, is_shorten=True):
	rle = [WorldUtils.cells_2_rle(world.world_cells[c], is_shorten) for c in CHANNEL]
	params = [WorldUtils.params_to_data(world.params[k]) for k in KERNEL]
	data = {'code': world.names[0], 'name': world.names[1], 'cname': world.names[2]}