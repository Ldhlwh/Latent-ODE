import torch
import torch.nn as nn
from torchdiffeq import odeint

class OR_ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(OR_ODE_Func, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['LO_hidden_size'], self.param['OF_layer_dim'])
		self.tanh = nn.Tanh()
		self.hidden_layer2 = nn.Linear(self.param['OF_layer_dim'], self.param['OF_layer_dim'])
		self.tanh2 = nn.Tanh()
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['LO_hidden_size'])
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.tanh(x)
		x = self.hidden_layer2(x)
		x = self.tanh2(x)
		x = self.output_layer(x)	
		return x
		
class GRU(nn.Module):
	
	def __init__(self, param):
		super(GRU, self).__init__()
		self.param = param
		self.update_gate = nn.Sequential(
			nn.Linear(self.param['OR_hidden_size'] + 1, self.param['GRU_unit']),
			nn.Tanh(),
			nn.Linear(self.param['GRU_unit'], self.param['LO_hidden_size']),
			nn.Sigmoid())
		self.reset_gate = nn.Sequential(
			nn.Linear(self.param['OR_hidden_size'] + 1, self.param['GRU_unit']),
			nn.Tanh(),
			nn.Linear(self.param['GRU_unit'], self.param['LO_hidden_size']),
			nn.Sigmoid())
		self.new_state_net = nn.Sequential(
			nn.Linear(self.param['OR_hidden_size'] + 1, self.param['GRU_unit']),
			nn.Tanh(),
			nn.Linear(self.param['GRU_unit'], self.param['OR_hidden_size']))
	
	def forward(self, mean, std, x, mask):
		y = torch.cat([mean, std, x], -1)
		
		update = self.update_gate(y)
		reset = self.reset_gate(y)
		y_concat = torch.cat([mean * reset, std * reset, x], -1)
		
		new_state_output = self.new_state_net(y_concat)
		new_mean = (1 - update) * new_state_output[:, 0:self.param['LO_hidden_size']] + update * mean
		new_std = (1 - update) * new_state_output[:, self.param['LO_hidden_size']:] + update * std
		
		final_mean = mask.reshape(-1, 1) * new_mean + (1 - mask).reshape(-1, 1) * mean
		final_std = mask.reshape(-1, 1) * new_std + (1 - mask).reshape(-1, 1) * std

		return final_mean, final_std.abs()
		
		
class ODE_RNN(nn.Module):
	
	def __init__(self, param):
		super(ODE_RNN, self).__init__()
		self.param = param
		self.ode_func = OR_ODE_Func(param)
		self.gru = GRU(param)
		self.mean0 = torch.zeros(self.param['batch_size'], self.param['LO_hidden_size'], device = self.param['device'])
		self.std0 = torch.zeros(self.param['batch_size'], self.param['LO_hidden_size'], device = self.param['device'])
	
	def forward(self, input_tuple):
		b = input_tuple[0]	# (batch_size, num_time_points, 2)
		m = input_tuple[1]	# (batch_size, num_time_points)
		train_m = input_tuple[2]	# (batch_size, num_time_points)
		#test_m = input_tuple[3]	# (batch_size, num_time_points)
		
		mean_ode = odeint(self.ode_func, self.mean0, torch.tensor([self.param['time_horizon'], b[0, -1, 0]], device = self.param['device']), rtol = self.param['rtol'], atol = self.param['atol'])[1]
		mean, std = self.gru(mean_ode, self.std0, b[:, -1, 1].reshape(-1, 1), train_m[:, -1])
		
		for i in range(b.shape[1] - 2, -1, -1):
			mean_ode = odeint(self.ode_func, mean, torch.tensor([b[0, i + 1, 0], b[0, i, 0]], device = self.param['device']), rtol = self.param['rtol'], atol = self.param['atol'])[1]
			mean, std = self.gru(mean_ode, std, b[:, i, 1].reshape(-1, 1), train_m[:, i])
			
		return mean, std