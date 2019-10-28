import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from ODE_RNN import ODE_RNN

class ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(ODE_Func, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['LO_hidden_size'], self.param['OF_layer_dim'])
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['LO_hidden_size'])
		self.tanh = nn.Tanh()
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.output_layer(x)
		x = self.tanh(x)
		return x

class Latent_ODE(nn.Module):

	def __init__(self, param):
		super(Latent_ODE, self).__init__()
		self.param = param
		self.ode_rnn = ODE_RNN(self.param)
		self.gmu_hidden = nn.Linear(self.param['OR_hidden_size'], self.param['g_hidden_dim'])
		self.gmu_output = nn.Linear(self.param['g_hidden_dim'], self.param['LO_hidden_size'])
		self.gsigma_hidden = nn.Linear(self.param['OR_hidden_size'], self.param['g_hidden_dim'])
		self.gsigma_output = nn.Linear(self.param['g_hidden_dim'], self.param['LO_hidden_size'])
		self.ode_func = ODE_Func(param)
		self.output_hidden = nn.Linear(self.param['LO_hidden_size'], self.param['LO_hidden_dim'])
		self.output_output = nn.Linear(self.param['LO_hidden_dim'], 1)
	
	def forward(self, input):
		x = input	# (batch_size, obs_points, 2)
		train_x = x[:, 0:self.param['obs_points'], :]
		test_x = x[:, self.param['obs_points'] - 1:, :]
		z0p = self.ode_rnn(train_x)
		
		mu = self.gmu_hidden(z0p)
		mu = self.gmu_output(mu)
		sigma = self.gsigma_hidden(z0p)
		sigma = self.gsigma_output(sigma)
		
		z0 = torch.normal(mu, sigma)
		z_out = odeint(self.ode_func, z0, test_x[0, :, 0])
		
		output = torch.zeros(test_x.shape[1] - 1)
		for i in range(1, test_x.shape[1]):
			out = self.output_hidden(z_out[i])
			out = self.output_output(out)
			output[i - 1] = out
		
		return output