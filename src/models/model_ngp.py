import torch
from torch import nn
from torch.nn import functional as F

from .layers import Linear

from .encodings import HashGridEncoder

class PositionalDecoder(nn.Module):
	def __init__(self,
		layers, embedding_dim, min_res=16, max_res=28):
		super().__init__()
		self.encoder = HashGridEncoder(layers=layers,
			dim=embedding_dim, min_res=min_res, max_res=max_res)
		self.net = nn.Sequential(
				Linear(embedding_dim*layers, 128), nn.ReLU(),
				Linear(128, 128), nn.ReLU(),
				Linear(128, 3)
			)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		x = self.encoder(x)
		x = self.net(x)
		return x