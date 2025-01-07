import abc

import rlp
from rlp.sedes import big_endian_int, text, List

RLP_DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}
# default is 2 dimensional
DIM = 2

def rle_to_cells(data: str) -> np.ndarray:
	stacks = [[] for dim in range(DIM)]
	last, count = '', ''
	# list of delimiters
	delims = list(DIM_DELIM.values())
	data = data.rstrip("!") + DIM_DELIM[DIM - 1]
	for ch in data:
		if ch.isdigit(): count += ch
		elif ch in 'pqrstuvwxy@': last = ch
		else:
			if last + ch not in delims:
				World.add_cells(stacks[0], World.chr_to_cell(last + ch) / 255, count, is_repeat=True)
			else:
				dim = delims.index(last + ch)
				for d in range(dim):
					World.add_cells(stacks[d + 1], stacks[d], count, is_repeat=False)
					stacks[d] = []
			last, count = '', ''

		cells = stacks[DIM - 1]
		max_lens = [0 for dim in range(DIM)]
		recur_get_max_lens(0, cells, max_lens)
		recur_cubify(0, cells, max_lens)
		return np.asarray(cells)

def cells_2_rle(cells: np.ndarray) -> str:
	''' Convert `np.ndarray` layouted entity into RLE data

	Does add '!' at the ned, converts only commands
		(idea behind this is that it insures that you know what you're doing)

	cells : np.ndarray : The ndarray format of the entity data core
	Returns : str -> In the RLE format, no metadata and no comments
	'''
	if isinstance(cells, np.ndarray):
		values = cells.tolist()
		if is_shorten:
			rle_groups = recur_drill_list(0, values, lambda row: [(len(list(g)), World.cell_to_char(v).strip()) for v, g in itertools.groupby(row)])
			data = recursive_joiner(0, rle_groups, lambda: row: [(str(n) if n > 1 else '') + c for n, c in row])
		else:
			data = recursive_joiner(0, values, lambda row: [World.cell_to_char(v) for v in row])
		return data + "!"

def recursive_joiner(dim, lists, row_func):
	if dim < DIM - 1:
		return RLP_DIM_DELIM[DIM - 1 - dim].join(recursive_joiner(dim + 1, e, row_func) for e in lists)
	else:
		return RLP_DIM_DELIM[DIM - 1 - dim].join(row_func(lists))

def formatOut(sequence):
	result = []
	for item in sequence:
		if (item[1] == 1):
			result.append(item[0])
		else:
			result.append(str(item[1]) + item[0])
	return "".join(result)
	

def from_data(cls, data):
	d_cls = cls()
	d_cls.names = [data.get('code', ''), data.get('name', ''), data.get('cname', '')]
	d_cls.params = None
	