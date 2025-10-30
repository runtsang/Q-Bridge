import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """A lightweight fully‑connected layer with a single output neuron.

    The layer applies a linear transformation followed by a ``tanh`` non‑linearity.
    It accepts an iterable of parameters (``thetas``) and returns the mean
    activation as a NumPy array, matching the original interface.  The implementation
    is fully GPU‑aware through PyTorch and can be embedded in larger pipelines.

    Args:
        n_features: Number of input features (default 1).
        bias: Whether to use a bias term.
    """
    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Forward pass returning a tensor of activations."""
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean activation as a NumPy array for compatibility."""
        return self.forward(thetas).mean().detach().numpy()

__all__ = ["FCL"]
