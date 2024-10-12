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

Install JAX:

```
$ pip install jax-metal ml_dtypes==0.2.0 jax==0.4.26 jaxlib==0.4.26
```

Using MPS (Metal Performance Shaders)

```
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mps = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"MPS Available: {torch.backends.mps.is_available()}")

# Example usage:
# scalar
scalar = torch.tensor(7, device = mps)
scalar
```
