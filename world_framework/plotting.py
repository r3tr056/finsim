import matplotlib.pyplot as plt
import numpy as np

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

