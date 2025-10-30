"""Enhanced fully connected layer with dropout and batch normalization.

This module defines a PyTorch implementation of a fully connected layer
with optional dropout and batch‑normalisation.  The class can be used
as a drop‑in replacement for the original simple layer while providing
additional regularisation and a more expressive forward pass.
"""

import torch
from torch import nn
import numpy as np


class FCL(nn.Module):
    """
    A fully connected neural network layer with optional dropout and
    batch‑normalisation.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_hidden : int, optional
        Size of the hidden layer.  If None, a single linear layer is used.
    n_output : int, optional
        Number of output units.  Defaults to 1.
    dropout : float, optional
        Drop‑out probability.  If 0.0 no dropout is applied.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int | None = None,
        n_output: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []

        if n_hidden is None:
            # Simple linear layer
            layers.append(nn.Linear(n_features, n_output))
        else:
            # Two‑layer MLP with dropout and batch norm
            layers.append(nn.Linear(n_features, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(n_hidden, n_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def run(self, thetas: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Run the network on a batch of inputs and return the mean output.

        Parameters
        ----------
        thetas : array‑like
            Input data of shape (batch, features).

        Returns
        -------
        np.ndarray
            Mean output value over the batch.
        """
        if isinstance(thetas, np.ndarray):
            thetas = torch.from_numpy(thetas).float()
        else:
            thetas = thetas.float()

        with torch.no_grad():
            out = self.forward(thetas)
        # Compute mean over batch dimension
        return out.mean(dim=0).cpu().numpy()


__all__ = ["FCL"]
