import numpy as np
import torch
from torch import nn
from typing import Iterable

class QuantumFullyConnectedLayer(nn.Module):
    """
    Classical surrogate of a variational quantum fully‑connected layer.
    Accepts a list of scalars ``thetas`` that are interpreted as layer
    weights.  The module supports an arbitrary number of input features
    and a configurable output dimension, and exposes a ``run`` method
    compatible with the original seed.  The implementation is deliberately
    simple so that it can be used as a drop‑in replacement for the
    quantum version during debugging or as a baseline.
    """
    def __init__(self, n_features: int = 1, out_features: int = 1,
                 bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic a forward pass of a quantum layer.
        * ``thetas`` are interpreted as a flat list of weights for the
          underlying linear transformation.
        * The method returns a NumPy array containing the mean
          activation over the input dimension, matching the
          signature of the original reference.
        """
        # Convert thetas to a tensor and reshape to match the linear weight shape
        theta_arr = torch.as_tensor(list(thetas), dtype=torch.float32)
        expected_size = self.linear.weight.numel() + (self.linear.bias.numel() if self.linear.bias is not None else 0)
        if theta_arr.numel()!= expected_size:
            raise ValueError(f"Expected {expected_size} parameters, got {theta_arr.numel()}")
        # Unpack parameters
        weight_end = self.linear.weight.numel()
        weight_vals = theta_arr[:weight_end].view_as(self.linear.weight)
        bias_vals = theta_arr[weight_end:] if self.linear.bias is not None else None

        # Assign weights
        with torch.no_grad():
            self.linear.weight.copy_(weight_vals)
            if bias_vals is not None:
                self.linear.bias.copy_(bias_vals)

        # Forward pass on a dummy batch of ones
        dummy_input = torch.ones((1, self.linear.in_features))
        out = self.linear(dummy_input)
        out = torch.tanh(out)  # mimic quantum activation
        out = self.dropout(out)
        return out.mean(dim=0).detach().numpy()

__all__ = ["QuantumFullyConnectedLayer"]
