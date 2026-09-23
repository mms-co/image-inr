import torch
import numpy as np

def generate(size):
	assert isinstance(size, tuple)
	assert len(size) == 2
	width, height = size
	pos_img = torch.zeros(height, width, 2)
	for x in np.arange(width):
		pos_img[:,x,0] = x
	for y in np.arange(height):
		pos_img[y,:,1] = y
	pos_map = pos_img.reshape(-1, 2)
	return pos_map