"""Fully connected neural network layer with batched parameter support.

The layer mimics a quantum fully connected layer by applying a linear mapping
followed by a hyperbolic tangent non‑linearity.  It accepts a batch of
parameter vectors and returns the corresponding activations as a NumPy array.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence


class FullyConnectedLayer(nn.Module):
    """A classical fully‑connected layer that can process batched parameters.

    Parameters
    ----------
    n_features : int
        Number of input features (i.e. dimensionality of each parameter vector).
    n_outputs : int, default=1
        Number of output units.  The original seed used a single output; this
        implementation generalises to an arbitrary number.
    """

    def __init__(self, n_features: int, n_outputs: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)

    def run(self, thetas: Iterable[Sequence[float]]) -> np.ndarray:
        """Apply the layer to a batch of theta vectors.

        Parameters
        ----------
        thetas
            Iterable of parameter vectors.  Each vector must have length
            ``n_features``.  The function accepts either a list of lists,
            a NumPy array, or a PyTorch tensor.

        Returns
        -------
        np.ndarray
            Shape ``(batch, n_outputs)`` containing the tanh activation of the
            linear transformation.
        """
        # Ensure a 2‑D tensor of shape (batch, n_features)
        theta_arr = torch.as_tensor(
            thetas, dtype=torch.float32, device=self.linear.weight.device
        )
        if theta_arr.ndim == 1:
            theta_arr = theta_arr.unsqueeze(0)

        # Forward pass
        output = torch.tanh(self.linear(theta_arr))
        return output.detach().cpu().numpy()

    def parameters(self):
        """Return the underlying trainable parameters."""
        return self.linear.parameters()


__all__ = ["FullyConnectedLayer"]
