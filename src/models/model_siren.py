import torch
from torch import nn

from .layers import *


class PositionalDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			SIRENLinear(2, 128, omega=30, is_first=True),
			SIRENLinear(128, 128, omega=30.),
			SIRENLinear(128, 128, omega=30.),
			SIRENLinear(128, 128, omega=30.),
			Linear(128, 1)
		)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		x = self.net(x)
		return x