import torch
from torch import nn
from torch.nn import functional as F

from .layers import Linear
from .encodings import NeRFEncoder
from .encodings import RandFEncoder

class PositionalDecoder(nn.Module):
	def __init__(self, frequencies):
		super().__init__()
		# self.encoder = RandFEncoder(seed=0,
		# 	frequencies=frequencies, mul=1)
		self.encoder = NeRFEncoder(frequencies=frequencies)
		in_signals = frequencies * 4
		self.net = nn.Sequential(
			Linear(in_signals, 128), nn.ReLU(),
			Linear(128, 128), nn.ReLU(),
			Linear(128, 1)
		)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		x = self.encoder(x)
		x = self.net(x)
		x = 4 * x - 2
		x = F.sigmoid(x)
		return x