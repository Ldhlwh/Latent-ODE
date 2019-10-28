import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(ODE_Func, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['OR_hidden_size'], self.param['OF_layer_dim'])
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['OR_hidden_size'])
		self.tanh = nn.Tanh()
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.output_layer(x)
		x = self.tanh(x)
		return x

class ODE_RNN(nn.Module):
	
	def __init__(self, param):
		super(ODE_RNN, self).__init__()
		self.param = param
		self.ode_func = ODE_Func(self.param)
		self.rnn_cell = nn.GRU(input_size = 1, hidden_size = param['OR_hidden_size'], batch_first = True)
		self.h0 = torch.zeros(self.param['OR_hidden_size'])
		
	def forward(self, input):
		x = input	# (batch_size, obs_points, 2)
		hp = odeint(self.ode_func, self.h0, torch.tensor([0.0, x[0, 0, 0]]))
		_, h = self.rnn_cell(x[:, 0, 1].reshape(self.param['batch_size'], 1, 1))
		for i in range(1, x.shape[1]):
			hp = odeint(self.ode_func, h, torch.tensor([x[0, i - 1, 0], x[0, i, 0]]))
			_, h = self.rnn_cell(x[:, i, 1].reshape(self.param['batch_size'], 1, 1), hp[1].reshape(self.param['batch_size'], 1, self.param['OR_hidden_size']))
		return h