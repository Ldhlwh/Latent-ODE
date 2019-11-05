import torch
import torch.nn as nn
from torchdiffeq import odeint
from ODE_RNN import ODE_RNN
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

class LO_ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(LO_ODE_Func, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['LO_hidden_size'], self.param['OF_layer_dim'])
		self.hidden_tanh = nn.Tanh()
		self.hidden_layer2 = nn.Linear(self.param['OF_layer_dim'], self.param['OF_layer_dim'])
		self.hidden_tanh2 = nn.Tanh()
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['LO_hidden_size'])
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.hidden_tanh(x)
		x = self.hidden_layer2(x)
		x = self.hidden_tanh2(x)
		x = self.output_layer(x)
		return x

class Latent_ODE(nn.Module):

	def __init__(self, param):
		super(Latent_ODE, self).__init__()
		self.param = param
		self.ode_func = LO_ODE_Func(param)
		self.ode_rnn = ODE_RNN(param)
		self.output_output = nn.Sequential(
			nn.Linear(self.param['LO_hidden_size'], 1),
		)
		self.mse_func = nn.MSELoss()
	
	def forward(self, input, kl_coef):
		b, m, train_m, test_m = input
		
		mean, std = self.ode_rnn(input)	# (batch_size, LO_hidden_size) * 2
		
		d = Normal(torch.tensor([0.0], device = self.param['device']), torch.tensor([1.0], device = self.param['device']))
		r = d.sample(mean.shape).squeeze(-1)
		z0 = mean + r * std
		
		z_out = odeint(self.ode_func, z0, b[0, :, 0], rtol = self.param['rtol'], atol = self.param['atol']) # (num_time_points, batch_size, LO_hidden_size)	
		z_out = z_out.permute(1, 0, 2)
		output = self.output_output(z_out).squeeze(2)

		z0_distr = Normal(mean, std)
		kl_div = kl_divergence(z0_distr, Normal(torch.tensor([0.0], device = self.param['device']), torch.tensor([1.0], device = self.param['device'])))
		kl_div = kl_div.mean(axis = 1)
		
		masked_output = output[test_m.bool()].reshape(self.param['batch_size'], (self.param['total_points'] - self.param['obs_points']))
		target = b[:, :, 1][test_m.bool()].reshape(self.param['batch_size'], (self.param['total_points'] - self.param['obs_points']))
		
		gaussian = Independent(Normal(loc = masked_output, scale = self.param['obsrv_std']), 1)
		log_prob = gaussian.log_prob(target)
		likelihood = log_prob / output.shape[1]
		
		loss = -torch.logsumexp(likelihood - kl_coef * kl_div, 0)
		mse = self.mse_func(masked_output, target)
		
		return loss, mse, masked_output
		