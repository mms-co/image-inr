import torch
from torch import nn
from .layers import Linear

# This model is built for MNIST
# To change this in RGB-compatible, make the last linear output 3 features
class PositionalDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			Linear(2, 512, activation="leakyrelu"), nn.LeakyReLU(.1),
			Linear(512, 512, activation="leakyrelu"), nn.LeakyReLU(.1),
			Linear(512, 1024, activation="leakyrelu"), nn.LeakyReLU(.1),
			Linear(1024, 512), nn.ReLU(),
			Linear(512, 512), nn.ReLU(),
			Linear(512, 256), nn.ReLU(),
			Linear(256, 256), nn.ReLU(),
			Linear(256, 1), nn.Sigmoid()
			)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		x = self.net(x)
		# x = x * 4 - 2
		# x = nn.Sigmoid()(x)
		return x