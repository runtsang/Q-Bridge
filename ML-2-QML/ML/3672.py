"""Hybrid fully connected layer combining classical RBF kernel and linear transform."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class HybridFCL(nn.Module):
    """
    A classical hybrid layer: maps input through an RBF kernel to a set of
    support vectors, then applies a learnable linear transform and a tanh
    non‑linearity.
    """

    def __init__(self, n_features: int = 1, gamma: float = 1.0, n_support: int = 8) -> None:
        super().__init__()
        self.gamma = gamma
        # fixed support vectors – could be learned, but kept constant for demo
        self.register_buffer("support", torch.randn(n_support, n_features))
        self.linear = nn.Linear(n_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the layer output for a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1) after tanh non‑linearity.
        """
        diff = x[:, None, :] - self.support[None, :, :]
        rbf = torch.exp(-self.gamma * (diff ** 2).sum(dim=-1, keepdim=True))
        out = self.linear(rbf)
        return torch.tanh(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper mimicking the original FCL interface. Converts an
        iterable of parameters into a single‑sample tensor and returns the
        output as a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable representing a single input vector.

        Returns
        -------
        np.ndarray
            NumPy array of shape (1,) containing the layer output.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        return self.forward(values).detach().numpy().reshape(-1)


def FCL() -> HybridFCL:
    """Return an instance of the hybrid classical fully‑connected layer."""
    return HybridFCL()


__all__ = ["HybridFCL", "FCL"]
