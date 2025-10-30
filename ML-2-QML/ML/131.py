"""Enhanced fully connected layer with optional hidden block and dropout.

The class extends the original seed by adding a configurable hidden layer,
dropout for regularisation, and a convenience ``run`` method that accepts
an iterable of parameters and returns the network output as a NumPy array.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class FCL(nn.Module):
    """
    A classical feed‑forward layer that emulates a quantum fully‑connected
    layer.  The network consists of an optional hidden linear block followed
    by a ReLU, dropout and a final linear output.

    Parameters
    ----------
    n_features : int, default 1
        Size of the input feature vector.
    n_hidden : int, default 32
        Number of neurons in the hidden layer.  Set to 0 for a single‑layer
        linear model.
    dropout : float, default 0.1
        Drop‑out probability applied after the hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.dropout = dropout

        layers: Sequence[nn.Module] = []

        if n_hidden > 0:
            layers.append(nn.Linear(n_features, n_hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(n_hidden, 1))
        else:
            # Pure linear mapping
            layers.append(nn.Linear(n_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Apply the model to an iterable of scalar parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input values.  Each value is treated as a single
            feature.

        Returns
        -------
        np.ndarray
            Array of shape (len(thetas),) containing the model output.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            output = self.forward(values)
        return output.squeeze(-1).detach().cpu().numpy()

    def params(self) -> Sequence[torch.Tensor]:
        """Return a list of trainable parameters."""
        return list(self.parameters())


__all__ = ["FCL"]
