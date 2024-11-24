# Installing PyTorch

## Create a virtual environment and update pip:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

# Common Imports
## Install Torch:

```
pip install torch torchvision torchaudio
```


## Import Torch:
```
import torch
# Check the version
print(f"PyTorch version: {torch.__version__}")
```

## Other Common Imports:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn # Can also import the common abbreviation "nn" for "Neural Networks"
import torchvision # Base computer vision library
import torchtext # Base text and natural language processing library
from torchtext import datasets, models, transforms # Other components of TorchText (premade datasets, pretrained models and text transforms)
from torch.utils.data import DataLoader


```

## Device-Agnostic Code:

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
