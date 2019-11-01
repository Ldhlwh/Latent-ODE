import numpy as np
import torch
import torch.nn as nn
import time
from torchdiffeq import odeint_adjoint as odeint
from toy_batch import BatchGenerator
from make_batch_mask import *
from ODE_RNN import *
from toy_sample import sample
from torch.distributions.normal import Normal
from torch.distributions import Independent

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.system('rm -f fig/*')
os.system('rm -f test_fig/*')

param = {
	'resample': False,
	'time_horizon': 50,
	'period_min': 5,
	'period_max': 10,
	'num_seq': 800,
	'test_num_seq': 200,
	'num_point_in_seq': 100,
	'obs_points': 30,
	'total_points': 100,
	'batch_size': 50, 
	'figure_per_batch': 5,
	'sigma': 0.1,
	
	# ODE_Func
	'OF_layer_dim': 100,
	
	# ODE_RNN
	'OR_hidden_size': 20,
	
	# Latent_ODE
	'g_hidden_dim':100,
	'LO_hidden_size': 10,
	'LO_hidden_dim': 100,
	
	# Hyperparam
	'num_iter': 20,
	'lr': 1e-2,
	
	# ODE Solver
	'rtol': 1e-3,
	'atol': 1e-4,
	
	'device': torch.device('cpu'),
	'train_color': 'dodgerblue',
	'test_color': 'orange',
}
#if torch.cuda.is_available():
#	param['device'] = torch.device('cuda:3')
	
print('Train on', param['device'])

if param['resample']:
	sample(param, 'train')
	sample(param, 'test')

bg = BatchGenerator(param['batch_size'], 'train', param)
tbg = BatchGenerator(param['batch_size'], 'test', param)

model = ODE_RNN(param).to(param['device'])
optimizer = torch.optim.Adamax(model.parameters(), lr = param['lr'])
mse = torch.nn.MSELoss()

for iter in range(param['num_iter']):
	print('Iter: %d' % iter)
	model.train()
	print('\tTrain:')
	bg.rewind()
	bn = 0
	while bg.has_next_batch():
		tic = time.time()
		batch = bg.next_batch()
		
		b, m, train_m, test_m = make_batch_mask(batch, param)

		input_tuple = (b, m, train_m, test_m)
		#tec = time.time()
		#print('Batch got in %.2f sec' % (tec - tic))
		optimizer.zero_grad()
		output = model.forward(input_tuple)
		masked_output = output[test_m.bool()].reshape(param['batch_size'], (param['total_points'] - param['obs_points']))
		target = b[:, :, 1][test_m.bool()].reshape(param['batch_size'], (param['total_points'] - param['obs_points']))
		
		log_likelihood = torch.tensor(0.0)
		for i in range(masked_output.shape[0]):
			gaussian = Independent(Normal(masked_output[i], param['sigma']), 1)
			ll = gaussian.log_prob(target[i]) / masked_output.shape[1]
			log_likelihood += ll
		log_likelihood /= masked_output.shape[0]
		loss = -log_likelihood
		mse_loss = mse(masked_output, target)
		
		#tac = time.time()
		#print('Forward finished in %.2f sec' % (tac - tec))
		loss.backward()
		#tuc = time.time()
		#print('Backward fininshed in %.2f sec' % (tuc - tac))
		optimizer.step()
		toc = time.time()
		
		for k in range(param['figure_per_batch']):
			plt.clf()
			plt.plot(b[k, :, 0][m[k].bool()].detach(), b[k, :, 1][m[k].bool()].detach(), color = param['train_color'])
			plt.plot(b[k, :, 0][train_m[k].bool()].detach(), b[k, :, 1][train_m[k].bool()].detach(), '.', color = param['train_color'])
			plt.plot(b[k, :, 0][test_m[k].bool()].detach(), output[k][test_m[k].bool()].detach(), color = param['test_color'], marker = '.')
			#plt.plot(b[k, :, 0][test_m[k].bool()].detach(), masked_output.reshape(param['batch_size'], param['total_points'] - param['obs_points'])[k].detach(), '.')
			plt.savefig('fig/' + str(iter) + '_' + str(bn) + '_' + str(k) + '.jpg')
			
		print('\tBatch: %4d | LL: %f | MSE: %f | Time: %.2f sec' % (bn, log_likelihood.item(), mse_loss.item(), toc - tic))
		bn += 1
	
	
	model.eval()
	print('\tTest:')
	with torch.no_grad():
		tbg.rewind()
		bn = 0
		while tbg.has_next_batch():
			tic = time.time()
			test_batch = tbg.next_batch()
			b, m, train_m, test_m = make_batch_mask(test_batch, param)

			input_tuple = (b, m, train_m, test_m)
			#tec = time.time()
			#print('Batch got in %.2f sec' % (tec - tic))
			output = model.forward(input_tuple)
			masked_output = output[test_m.bool()].reshape(param['batch_size'], (param['total_points'] - param['obs_points']))
			target = b[:, :, 1][test_m.bool()].reshape(param['batch_size'], (param['total_points'] - param['obs_points']))
			
			log_likelihood = torch.tensor(0.0)
			for i in range(masked_output.shape[0]):
				gaussian = Independent(Normal(masked_output[i], param['sigma']), 1)
				ll = gaussian.log_prob(target[i]) / masked_output.shape[1]
				log_likelihood += ll
			log_likelihood /= masked_output.shape[0]
			loss = -log_likelihood
			mse_loss = mse(masked_output, target)
			
			toc = time.time()
			
			for k in range(param['figure_per_batch']):
				plt.clf()
				plt.plot(b[k, :, 0][m[k].bool()].detach(), b[k, :, 1][m[k].bool()].detach(), color = param['train_color'])
				plt.plot(b[k, :, 0][train_m[k].bool()].detach(), b[k, :, 1][train_m[k].bool()].detach(), '.', color = param['train_color'])
				plt.plot(b[k, :, 0][test_m[k].bool()].detach(), output[k][test_m[k].bool()].detach(), color = param['test_color'], marker = '.')
				plt.savefig('test_fig/' + str(iter) + '_' + str(bn) + '_' + str(k) + '.jpg')
			
			print('\tBatch: %4d | LL: %f | MSE: %f | Time: %.2f sec' % (bn, log_likelihood.item(), mse_loss.item(), toc - tic))
			bn += 1
