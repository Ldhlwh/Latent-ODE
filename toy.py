import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from toy_batch import BatchGenerator
from Latent_ODE import Latent_ODE

param = {
	'batch_size': 1,
	'obs_points': 30,
	'num_iter': 100,
	
	
	# ODE_Func
	'OF_layer_dim': 100,
	
	# ODE_RNN
	'OR_hidden_size': 20,
	
	# Latent_ODE
	'g_hidden_dim':100,
	'LO_hidden_size': 10,
	'LO_hidden_dim': 100,
	
}

bg = BatchGenerator(param['batch_size'])

model = Latent_ODE(param)
optimizer = torch.optim.Adamax(model.parameters(), lr = 0.01)
loss = torch.nn.MSELoss()

for iter in range(param['num_iter']):
	model.train()
	bg.rewind()
	ll = []
	while bg.has_next_batch():
		batch = bg.next_batch()
		
		#batch = torch.tensor(bg.next_batch(), dtype = torch.float32)
		optimizer.zero_grad()
		output = model.forward(batch)
		target = batch[0, param['obs_points']:, 1]
		final_loss = loss(output, target)
		ll.append(final_loss.item())
		final_loss.backward()
		optimizer.step()
		print('Loss: %f' % final_loss.item())
	
	avg_loss = np.mean(ll)
	print('Iter: %4d | Loss: %f' % (iter, avg_loss))