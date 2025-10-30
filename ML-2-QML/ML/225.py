"""Fully connected layer with batch support, dropout, and bias."""
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Union

class FCL(nn.Module):
    """A fully‑connected neural network layer with optional bias, dropout, and Tanh activation.

    The class can be used as a drop‑in replacement for a single‑parameter quantum layer.
    It exposes a :py:meth:`run` method that accepts a batch of input thetas and returns
    the network output as a NumPy array.
    """

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        return self.activation(self.dropout(self.linear(x)))

    def run(
        self, thetas: Union[Sequence[float], Sequence[Sequence[float]]]
    ) -> np.ndarray:
        """
        Evaluate the layer on a batch of input thetas.

        Parameters
        ----------
        thetas : Sequence[float] or Sequence[Sequence[float]]
            If a 1‑D sequence, it is treated as a single sample.
            If a 2‑D sequence, each inner sequence is a sample.

        Returns
        -------
        np.ndarray
            The network output as a NumPy array of shape (batch, out_features).
        """
        theta_tensor = torch.as_tensor(thetas, dtype=torch.float32)
        if theta_tensor.dim() == 1:
            theta_tensor = theta_tensor.unsqueeze(0)
        out = self.forward(theta_tensor)
        return out.detach().cpu().numpy()
