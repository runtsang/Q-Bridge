"""
Hybrid fully‑connected layer for classical neural networks.

Provides:
  * A torch.nn.Module that accepts a list of parameters (thetas)
    and returns a scalar expectation value.
  * Automatic differentiation of the expectation w.r.t. thetas.
  * A small training helper that runs one gradient‑descent step.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn, optim
import numpy as np


class FCLHybrid(nn.Module):
    """
    A classical fully‑connected layer that mimics a quantum layer.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features (i.e. the length of the theta vector).
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Compute the expectation value for a given set of parameters.

        The forward pass is simply a tanh‑activated linear transform
        followed by a mean over the output dimension, which is a common
        toy objective for quantum‑inspired layers.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter vector of length ``n_features``.

        Returns
        -------
        torch.Tensor
            Scalar expectation value.
        """
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32, requires_grad=True)
        out = torch.tanh(self.linear(theta_tensor))
        return out.mean()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Public API that mirrors the quantum seed.

        Returns a NumPy array containing the expectation value.
        """
        with torch.no_grad():
            expectation = self.forward(thetas)
        return expectation.detach().numpy()

    def train_step(self, thetas: Iterable[float], target: float = 0.0) -> float:
        """
        Perform one gradient‑descent step on the parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Current parameter vector.
        target : float, optional
            Desired target value for the expectation.

        Returns
        -------
        float
            Loss value after the step.
        """
        self.optimizer.zero_grad()
        loss = (self.forward(thetas) - target) ** 2
        loss.backward()
        self.optimizer.step()
        return loss.item()


__all__ = ["FCLHybrid"]
