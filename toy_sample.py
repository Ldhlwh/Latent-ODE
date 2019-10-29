import numpy as np
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''

def sample(param, type):
	if type == 'train':
		total_seq = param['num_seq']
	elif type == 'test':
		total_seq = param['test_num_seq']

	a = np.zeros((total_seq, param['total_points'], 2))
	
	div_time = 5 * param['obs_points'] / param['total_points']
	for i in range(total_seq):

		T = np.random.rand() / 2 + 0.5
		x1 = np.random.rand(param['obs_points']) * div_time
		x2 = np.random.rand(param['total_points'] - param['obs_points']) * (5 - div_time) + div_time
		x = np.concatenate((x1, x2))
		x.sort()
		a[i, :, 0] = x
		y = np.cos(x / (T / (2 * np.pi)))
		y = np.random.normal(y, 0.1)
		a[i, :, 1] = y
		'''
		if i < 10:
			plt.clf()
			plt.plot(a[i, :, 0], a[i, :, 1])
			plt.savefig('plot' + str(i) + '.jpg')
		'''
	if type == 'train':
		np.save('toy_data.npy', np.float32(a))
	elif type == 'test':
		np.save('toy_test.npy', np.float32(a))

'''
param = {
	'num_seq': 100,
	'total_points': 100,
	'obs_points': 30,
	}
	
sample(param)
'''