import torch
from torch import nn
import numpy as np

class NeRFEncoder(nn.Module):
	def __init__(self, frequencies):
		super().__init__()
		weights = np.pi * (2. ** torch.arange(frequencies))
		self.register_buffer("weights", weights, persistent=False)

	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		scaled_x = x.unsqueeze(-1) * self.weights.unsqueeze(0)
		proj = scaled_x.view(x.shape[0], -1)
		sin_vect = torch.sin(proj)
		cos_vect = torch.cos(proj)
		fourier = torch.cat((sin_vect, cos_vect), dim=1)
		return fourier