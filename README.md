# Image INR
An educational project implementing different neural networks for Implicit Neural Represenation (INR) for images.

---
## Features
- Different models
	* A regular MLP
	* SIREN
	* Fourier encoded
	* NGP
- Different layers and features
	* Linear with improved initialisation
	* Guassian Fourier encoding
	* NeRF
	* Hash Grids
- Supports JPG/PNG image formats
- Performance evaluation based on MSE and PSRN

## Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

## Installation
```bash
git clone https://github.com/mms-co/image-inr
cd image-inr
pip install -r requirements.txt
```

## Usage
An example model is already configured inside `main.py` ready to train and evaluate. To add images, put them in the `data/` folder and refer to the image in `main.py`. Some results on the various models documented in `docs/`.
```bash
python main.py
```

## License
Distributed under the MIT License. See `LICENSE` for more information.