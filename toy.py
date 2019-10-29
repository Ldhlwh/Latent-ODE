import numpy as np
import torch
import torch.nn as nn
import time
from torchdiffeq import odeint_adjoint as odeint
from toy_batch import BatchGenerator
from make_batch_mask import *
from Latent_ODE import Latent_ODE
from toy_sample import sample

param = {
	'resample': False,
	'num_seq': 800,
	'test_num_seq': 200,
	'num_point_in_seq': 100,
	'obs_points': 30,
	'total_points': 100,
	'batch_size': 50, 
	
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
	'lr': 1e-4,
	
	'device': torch.device('cpu'),
}
if torch.cuda.is_available():
	param['device'] = torch.device('cuda:3')
	
print('Train on', param['device'])

if param['resample']:
	sample(param, 'train')
	sample(param, 'test')

bg = BatchGenerator(param['batch_size'], 'train')
tbg = BatchGenerator(param['batch_size'], 'test')

model = Latent_ODE(param).to(param['device'])
optimizer = torch.optim.RMSprop(model.parameters(), lr = param['lr'])
loss_func = torch.nn.MSELoss()

for iter in range(param['num_iter']):
	print('Iter: %d' % iter)
	model.train()
	print('\tTrain:')
	bg.rewind()
	ll = []
	bn = 0
	while bg.has_next_batch():
		tic = time.time()
		batch = bg.next_batch()
		b_train, m_train = make_batch_mask(batch[:, 0:param['obs_points'], :], param)
		b_test, m_test = make_batch_mask(batch[:, param['obs_points']:, :], param)
		t0_test = b_train[0, -1, 0]
		input_tuple = (b_train, m_train, b_test, m_test, t0_test)
		
		#tec = time.time()
		#print('Batch got in %.2f sec' % (tec - tic))
		optimizer.zero_grad()
		output = model.forward(input_tuple)
		masked_output = output[m_test.bool()]
		target = torch.tensor(batch[:, param['obs_points']:, 1].flatten(), device = param['device'])
		
		loss = loss_func(masked_output, target)
		ll.append(loss.item())
		#tac = time.time()
		#print('Forward finished in %.2f sec' % (tac - tec))
		loss.backward()
		optimizer.step()
		toc = time.time()
		print('\tBatch: %4d | Loss: %f | Time: %.2f sec' % (bn, loss.item(), toc - tic))
		bn += 1
		
	train_avg_loss = np.mean(ll)
	
	model.eval()
	print('\tTest:')
	with torch.no_grad():
		tbg.rewind()
		ll = []
		bn = 0
		while tbg.has_next_batch():
			tic = time.time()
			test_batch = tbg.next_batch()
			b_train, m_train = make_batch_mask(test_batch[:, 0:param['obs_points'], :], param)
			b_test, m_test = make_batch_mask(test_batch[:, param['obs_points']:, :], param)
			t0_test = b_train[0, -1, 0]
			input_tuple = (b_train, m_train, b_test, m_test, t0_test)
			
			output = model.forward(input_tuple)
			masked_output = output[m_test.bool()]
			target = torch.tensor(batch[:, param['obs_points']:, 1].flatten(), device = param['device'])
			loss = loss_func(masked_output, target)
			ll.append(loss.item())
			
			toc = time.time()
			print('\tBatch: %4d | Loss: %f | Time: %.2f sec' % (bn, loss.item(), toc - tic))
			bn += 1
			
		test_avg_loss = np.mean(ll)
		print('Train Loss: %f | Test Loss: %f\n' % (train_avg_loss, test_avg_loss))
		