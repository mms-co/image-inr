import torch
from torch import nn

class Linear(nn.Module):
	def __init__(self, in_features, out_features, activation="relu"):
		super().__init__()
		self.linear = nn.Linear(in_features, out_features)

		with torch.no_grad():
			# He initialisation for hidden layers
			if activation == "relu":
				nn.init.kaiming_normal_(self.linear.weight,
					mode="fan_in", nonlinearity="relu")
			elif activation == "leakyrelu":
				nn.init.kaiming_normal_(self.linear.weight,
					mode="fan_in", a=0.1, nonlinearity="leaky_relu")
			nn.init.zeros_(self.linear.bias)
			# Assumes last layer, uses Xavier
			# (He not compatible for Sigmoid/Tanh)
			if out_features <= 3:
				nn.init.xavier_uniform_(self.linear.weight)
	def forward(self, x):
		x = self.linear(x)
		return x