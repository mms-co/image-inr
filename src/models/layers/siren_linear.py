import torch
from torch import nn
import numpy as np

class SIRENLinear(nn.Module):
	def __init__(self, in_features, out_features, omega, is_first=False):
		super().__init__()
		self.linear = nn.Linear(in_features, out_features)
		self.omega = omega

		with torch.no_grad():
			if is_first:
				bounds = 1. / in_features
				self.linear.weight.uniform_(-bounds, bounds)
				self.linear.bias.uniform_(-bounds, bounds)
			else:
				bounds = np.sqrt(6. / in_features) / omega
				self.linear.weight.uniform_(-bounds, bounds)
				nn.init.zeros_(self.linear.bias)
	def forward(self, x):
		x = self.linear(x)
		x = torch.sin(self.omega * x)
		return x