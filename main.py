import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.models import *

from src.utils.reader import read_image, read_mnist
from src.utils.generate_image_pos import generate

C_DIR = os.path.dirname(os.path.abspath(__file__))

PATH = C_DIR + "/data/imagenet-kingsnake.jpg"


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

image, size = read_image(PATH, dtype=np.float32)
height, width = size
image = torch.tensor(image)
image = image.view(width*height, -1)

model = NGPDecoder(layers=12, embedding_dim=2, min_res=16, max_res=512).to(device)
model = torch.compile(model)

epochs = 500
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.HuberLoss()

# Normalise input coordinates into [0,1]
pos_map = generate(size).to(torch.float32)
pos_map /= torch.tensor([width, height])
model.train()

""" if batching isn't needed
for epoch in np.arange(epochs):
	print(f"Training epoch {epoch+1}/{epochs}")
	optimizer.zero_grad()
	res = model(pos_map.to(device))
	loss = criterion(res, image.to(device))
	loss.backward()
	optimizer.step()
"""

# Randomise ordering
permutation = torch.randperm(pos_map.shape[0], device=device).to('cpu')
shuf_pos = pos_map[permutation]
shuf_img = image[permutation]
# Allocate and train on batches
# Note that batch_size must divide width*height
batch_size = 62500
batch_pos = shuf_pos.view(-1, batch_size, 2)
batch_img = shuf_img.view(-1, batch_size, 3)
for epoch in np.arange(epochs):
	for batch_p, batch_i in zip(batch_pos, batch_img):
		optimizer.zero_grad()
		res = model(batch_p.to(device))
		loss = criterion(res, batch_i.to(device))
		loss.backward()
		optimizer.step()



model.eval()
with torch.no_grad():
	res_img = model(pos_map.to(device)).to("cpu")
res_img = res_img.view(height, width, -1).detach().numpy()
image = image.view(height, width, -1).detach().numpy()
# Evaluate performance
res_img = np.clip(res_img, a_min=0., a_max=1.)
mse_val = np.mean((image - res_img)**2)
psnr_val = 20 * np.log10(1. / np.sqrt(mse_val))

print(f"MSE Loss: {mse_val}")
print(f"PSNR: {psnr_val}")


# View image
plt.imshow(res_img, cmap="gray")
plt.axis("off")
plt.show()