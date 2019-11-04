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
	
	#div_time = param['time_horizon'] * param['obs_points'] / param['total_points']
	for i in range(total_seq):

		T = np.random.rand() * (param['period_max'] - param['period_min']) + param['period_min']
		while True:
			x = np.random.rand(param['total_points']) * param['time_horizon']
			x.sort()
			if len(np.unique(x)) == param['total_points']:
				break
			
		a[i, :, 0] = x
		y = np.cos(x / (T / (2 * np.pi)))
		#y = np.random.normal(y, param['sigma'])
		a[i, :, 1] = y
		'''
		if i < 10:
			plt.clf()
			plt.plot(a[i, :, 0], a[i, :, 1], '.')
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