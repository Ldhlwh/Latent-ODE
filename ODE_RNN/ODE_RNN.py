import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class OR_ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(OR_ODE_Func, self).__init__()
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
		self.ode_func = OR_ODE_Func(self.param)
		self.rnn_cell = nn.GRUCell(input_size = 1, hidden_size = param['OR_hidden_size'])
		self.h0 = torch.zeros(self.param['batch_size'], self.param['OR_hidden_size'], device = self.param['device'])
		self.output_hidden = nn.Linear(self.param['OR_hidden_size'], self.param['OF_layer_dim'])
		self.output_output = nn.Linear(self.param['OF_layer_dim'], 1)
		self.output_tanh = nn.Tanh()
	
	def forward(self, input_tuple):
		b = input_tuple[0]	# (batch_size, num_time_points, 2)
		m = input_tuple[1]	# (batch_size, num_time_points)
		train_m = input_tuple[2]	# (batch_size, num_time_points)
		test_m = input_tuple[3]	# (batch_size, num_time_points)
		
		output = torch.zeros(b.shape[0], b.shape[1], device = self.param['device'])

		hp = odeint(self.ode_func, self.h0, torch.tensor([0.0, b[0, 0, 0]], device = self.param['device']), rtol = self.param['rtol'], atol = self.param['atol'])[1]
		out = self.output_hidden(hp)
		out = self.output_output(out)
		out = self.output_tanh(out)
		output[:, 0] = out.reshape(-1)
		htrain = self.rnn_cell((b[:, 0, 1] * train_m[:, 0]).reshape(-1, 1), hp)
		htest = self.rnn_cell((out.reshape(-1) * test_m[:, 0]).reshape(-1, 1), hp)
		h = torch.mul(train_m[:, 0].reshape(-1, 1), htrain) +torch.mul(test_m[:, 0].reshape(-1, 1), htest) + torch.mul((1 - m[:, 0]).reshape(-1, 1), hp)
		
		for i in range(1, b.shape[1]):
			hp = odeint(self.ode_func, h, b[0, i - 1:i + 1, 0], rtol = self.param['rtol'], atol = self.param['atol'])[1]
			out = self.output_hidden(hp)
			out = self.output_output(out)
			out = self.output_tanh(out)
			output[:, i] = out.reshape(-1)
			htrain = self.rnn_cell((b[:, i, 1] * train_m[:, i]).reshape(-1, 1), hp)
			htest = self.rnn_cell((out.reshape(-1) * test_m[:, i]).reshape(-1, 1), hp)
			h = torch.mul(train_m[:, i].reshape(-1, 1), htrain) + torch.mul(test_m[:, i].reshape(-1, 1), htest) + torch.mul((1 - m[:, i]).reshape(-1, 1), hp)
		
			
		return output