"""Enhanced fully connected layer implemented with PyTorch.

The class exposes a `run` method that accepts a flat list of parameters
and returns the mean activation of a two‑layer network.  The network
is fully trainable via standard PyTorch optimizers.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import numpy as np


class FCL(nn.Module):
    """Two‑layer fully connected network with ReLU activation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dim : int
        Number of neurons in the hidden layer.
    output_dim : int
        Dimensionality of the output.  Defaults to 1.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the network on a flat list of parameters.

        The parameters are interpreted as the input vector to the network.
        The method returns the mean of the final activations as a NumPy array.
        """
        # Convert the input list to a tensor of shape (1, input_dim)
        x = torch.tensor(
            list(thetas), dtype=torch.float32
        ).view(1, -1)
        with torch.no_grad():
            out = self.forward(x)
            mean_val = out.mean(dim=1)
        return mean_val.cpu().numpy()

__all__ = ["FCL"]
