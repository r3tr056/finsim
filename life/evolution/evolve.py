def evolve(cellular_automaton, timesteps, apply_rule, r=1):
	'''
	Evolves the cellular automation for the gives timesteps. Applies
	the gives function to each cell during the evolution. A cellular
	automation is represented as an array of arrays, or matrix. This
	function expects an array containing the initial timestep. The
	final result is a matrix, where the rows are equal to the 
	timesteps specified
	'''
	_, cols = cellular_automaton.shape
	array = np.zeros((timesteps, cols), dtype=cellular_automaton.dtype)
	array[0] = cellular_automaton

	def index_strides(arr, window_size):
		arr = np.concatenate((arr[-window_size//2+1:], arr, arr[:window_size//2]))
		shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
		strides = arr.strides + (arr.strides[-1],)
		return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

	for t in range(1, timesteps):
		cells = array[t - 1]
		strides = index_strides(np.arrange(len(cells)), 2*r + 1)
		neighbourhoods = cells[strides]
		array[t] = np.array([apply_rule(n, c, t) for c, n in enumerate(neighbourhoods)])
	return array

def bits_to_int(bits):
	total = 0
	for shift, j in enumerate(bits[::-1]):
		if j:
			total += 1 << shift
	return total

def int_to_bits(num, num_digits):
	converted = list(map(int, bin(num)[2:]))
	return np.pad(converted, (num_digits - len(converted), 0), 'constant')

