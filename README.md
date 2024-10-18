# Installing PyTorch, TensorFlow, and JAX on Apple Silicon

Create a virtual environment and update pip:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
```

Install Common Packages:
```
pip install pandas
pip install numpy
pip install matplotlib
```

Install Torch:

```
$ pip install torch torchvision torchaudio
```

Install TensorFlow:

```
$ pip install tensorflow tensorflow-macos tensorflow-metal
```

Common Imports
```
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check the version
print(f"PyTorch version: {torch.__version__}")

# Can also import the common abbreviation "nn" for "Neural Networks"
from torch import nn

# Base computer vision library
import torchvision


# Base text and natural language processing library
import torchtext

# Other components of TorchText (premade datasets, pretrained models and text transforms)
from torchtext import datasets, models, transforms


```

Device Agnostic Code:

```
# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")
```
