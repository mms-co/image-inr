import numpy as np

import imageio.v3 as iio

def read_image(path, dtype):
	img = iio.imread(path)
	img = img.astype(dtype) / 255.
	return img, img.shape[:-1]



def read_mnist(path, dtype):
	img = np.loadtxt(path, dtype=dtype)
	return img, (28,28)