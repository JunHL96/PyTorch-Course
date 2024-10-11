# Installing PyTorch, TensorFlow, and JAX on Apple Silicon

Create a virtual environment and update pip:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
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

Add a Device Configuration Block:

Step 1:
```
import torch

# Check if MPS is available, else fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Print the selected device for confirmation
print(f"Using device: {device}")

# A function to ensure tensors are moved to the correct device automatically
def move_to_device(tensor):
    return tensor.to(device)
```

Step 2:
```
# Example: Creating tensors and moving them to MPS (if available)
x = move_to_device(torch.rand(10000, 10000, dtype=torch.float32))
y = move_to_device(torch.rand(10000, 10000, dtype=torch.float32))

# Now all operations will happen on the MPS device
result = x * y
print(result)
```
