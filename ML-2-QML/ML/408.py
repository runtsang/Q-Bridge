"""Enhanced fully‑connected layer with dropout and multi‑output support."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCL(nn.Module):
    """
    A flexible fully‑connected layer that accepts a list of theta values,
    applies dropout, and maps them to a multi‑dimensional output.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input theta vector.
    output_dim : int, default 1
        Number of output neurons.
    dropout_rate : float, default 0.0
        Probability of zeroing an input element during training.
    device : str | torch.device, default 'cpu'
        Target device for tensors.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int = 1,
        dropout_rate: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass used internally by :meth:`run`.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameters.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (output_dim,).
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32, device=self.device)
        x = self.dropout(x)
        out = self.linear(x)
        return F.tanh(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Public API mirroring the original seed. Applies the forward pass
        and returns a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameters.

        Returns
        -------
        np.ndarray
            Numpy array of shape (output_dim,).
        """
        with torch.no_grad():
            out = self.forward(thetas)
        return out.cpu().numpy()


__all__ = ["FCL"]
