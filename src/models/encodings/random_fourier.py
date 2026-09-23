import torch
from torch import nn
import numpy as np

class RandFEncoder(nn.Module):
	def __init__(self, seed, frequencies, mul):
		super().__init__()
		torch.manual_seed(seed)
		weights = torch.randn(2, frequencies) * mul
		self.register_buffer("weights", weights, persistent=False)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		proj = x @ self.weights
		proj *= np.pi
		sin_vect = torch.sin(proj)
		cos_vect = torch.cos(proj)
		fourier = torch.cat((sin_vect, cos_vect), dim=1)
		return fourier