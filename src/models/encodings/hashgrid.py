import torch
from torch import nn

class HashGridEncoder(nn.Module):
	def __init__(self,
		layers, dim, min_res, max_res, log2_treshold=14):
		super().__init__()
		grid_factor = torch.exp((
			torch.log(torch.tensor(max_res)) -
			torch.log(torch.tensor(min_res))) /
			(layers - 1)
		)
		max_hash = 2**log2_treshold
		self.resolutions = [int(round(min_res * grid_factor.item()**i)) for i in range(layers)]
		self.embeddings = nn.ParameterList()
		for res in self.resolutions:
			verts = (res + 1)**2
			size = torch.min(torch.tensor([verts, max_hash]))
			embedding = nn.Parameter(torch.randn(size, dim) * 1e-4)
			self.embeddings.append(embedding)
		self.register_buffer("primes", torch.tensor([1, 2413867633], dtype=torch.int64), persistent=False)
		self.hash_treshold = max_hash
	def _hash_fn(self, coords, res):
		if (res+1)**2 < self.hash_treshold:
			return coords[:,0] + coords[:,1] * (res + 1)
		scaled_coords = coords * self.primes
		hash_idx = torch.bitwise_xor(scaled_coords[:,0], scaled_coords[:,1])
		return torch.remainder(hash_idx, self.hash_treshold)
	def forward(self, x):
		assert x.dim() == 2
		assert x.shape[1] == 2
		level_features = []
		for i, res in enumerate(self.resolutions):
			# Scale coordinates into resolution grid
			scaled_x = x * res
			# Get corner coodinates
			floor_x = torch.floor(scaled_x).to(torch.int64)
			floor_x = torch.clamp(floor_x, 0, res)
			ceil_x = floor_x + 1
			ceil_x = torch.clamp(ceil_x, 0, res)
			# Get all corner combinations (4)
			corner_00 = torch.stack([floor_x[:,0], floor_x[:,1]], dim=-1)
			corner_01 = torch.stack([floor_x[:,0], ceil_x[:,1]], dim=-1)
			corner_11 = torch.stack([ceil_x[:,0], ceil_x[:,1]], dim=-1)
			corner_10 = torch.stack([ceil_x[:,0], floor_x[:,1]], dim=-1)
			# Embedding for each corner
			idx_00 = self._hash_fn(corner_00, res)
			embed_00 = self.embeddings[i][idx_00]
			idx_01 = self._hash_fn(corner_01, res)
			embed_01 = self.embeddings[i][idx_01]
			idx_11 = self._hash_fn(corner_11, res)
			embed_11 = self.embeddings[i][idx_11]
			idx_10 = self._hash_fn(corner_10, res)
			embed_10 = self.embeddings[i][idx_10]
			# Interpolate
			dist = scaled_x - corner_00
			wx = dist[:,0].unsqueeze(-1)
			wy = dist[:,1].unsqueeze(-1)
			x_interpolate_bottom = embed_00 * (1-wx) + embed_10 * wx
			x_interpolate_top = embed_01 * (1-wx) + embed_11 * wx
			y_interpolate = x_interpolate_bottom * (1-wy) + x_interpolate_top * wy
			level_features.append(y_interpolate)
		return torch.cat(level_features, dim=-1)

if __name__=="__main__":
	h = HashGridEncoder(layers=12, dim=2, min_res=16, max_res=512)
	coords = torch.rand(50,2)
	res = h(coords)
	res = res.view(-1, 2)
	print(res.shape)