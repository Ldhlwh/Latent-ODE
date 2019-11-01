import numpy as np

class BatchGenerator:
	
	def __init__(self, batch_size, type, param):
		if type == 'train':
			self.data = np.load('toy_data.npy')
		elif type == 'test':
			self.data = np.load('toy_test.npy')
		np.random.shuffle(self.data)
		self.length = self.data.shape[0]
		self.cur = 0
		self.batch_size = batch_size
		self.check(type, param)
		
	def has_next_batch(self):
		return self.cur < self.length
		
	def next_batch(self):
		if self.cur + self.batch_size > self.length:
			self.cur = self.length
			return self.data[-self.batch_size : ]
		else:
			rtn = self.data[self.cur : self.cur + self.batch_size]
			self.cur += self.batch_size
			return rtn
			
	def rewind(self):
		self.cur = 0
		np.random.shuffle(self.data)
		
	def check(self, type, param):
		while self.has_next_batch():
			batch = self.next_batch
			b, m = make_batch_mask(batch, param)
			try:
				if type == 'train':
					assert(m.sum() == param['batch_size'] * param['obs_points'])
				elif type == 'test':
					assert(m.sum() == param['batch_size'] * (param['total_points'] - param['obs_points']))
			except AssertionError:
				print('Something is wrong with ' + type + ' data, please resample it')
				exit(1)
				
		self.rewind()
		print('Check on ' + type + ' data passed')