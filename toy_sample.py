import numpy as np
import pandas as pd
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''
a = np.zeros((1000, 100, 2))

for i in range(1000):

	T = np.random.rand() / 2 + 0.5
	x = np.random.rand(100) * 5
	x.sort()
	a[i, :, 0] = x
	y = np.cos(x / (T / (2 * np.pi)))
	y = np.random.normal(y, 0.1)
	a[i, :, 1] = y
	'''
	if i < 50:
		plt.clf()
		plt.plot(a[i, :, 0], a[i, :, 1])
		plt.savefig('plot' + str(i) + '.jpg')
	'''

np.save('toy_data.npy', a)