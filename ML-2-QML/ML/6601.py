"""Enhanced classical fully‑connected layer with trainable parameters and optional hidden units.

The class is a PyTorch ``nn.Module`` that can be used as a drop‑in replacement for the seed
implementation.  It supports:
- A single linear output unit or an optional hidden layer.
- A ``run`` method that accepts an iterable of angles and returns the forward pass
  as a NumPy array (useful for hybrid training).
- Full compatibility with PyTorch optimizers and loss functions.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


class FCL(nn.Module):
    """
    Fully‑connected layer with optional hidden layer.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_hidden : Optional[int]
        Size of a hidden layer. If ``None`` a single linear unit is used.
    """

    def __init__(self, n_features: int = 1, n_hidden: Optional[int] = None) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        if n_hidden is None:
            self.linear = nn.Linear(n_features, 1, bias=True)
        else:
            self.linear = nn.Sequential(
                nn.Linear(n_features, n_hidden, bias=True),
                nn.Tanh(),
                nn.Linear(n_hidden, 1, bias=True),
            )

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass given an iterable of angles.

        Parameters
        ----------
        thetas : Iterable[float]
            Input angles to be reshaped into a column vector.

        Returns
        -------
        torch.Tensor
            Output of the network as a column vector.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return self.linear(values)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that returns a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Input angles.

        Returns
        -------
        np.ndarray
            Forward output as a NumPy array.
        """
        with torch.no_grad():
            out = self.forward(thetas)
        return out.detach().cpu().numpy()


__all__ = ["FCL"]
