"""Hybrid fully-connected layer with classical deep network and quantum-inspired expectation.

This module implements a classical neural network that mirrors the
`EstimatorQNN` seed while preserving the original FCL API.  The network is a
fully‑connected feed‑forward model with multiple hidden layers and tanh activations.
It exposes a :py:meth:`run` method that accepts an iterable of parameters and
returns the network’s output as a NumPy array, matching the interface of the
quantum reference.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable


class HybridLayer(nn.Module):
    """
    Classical hybrid layer that encapsulates a multi‑layer feed‑forward network.
    The architecture follows the EstimatorQNN example: 2→8→4→1 with Tanh
    activations.  It can be used as a drop‑in replacement for the original
    FullyConnectedLayer while providing richer expressivity.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network with a NumPy‑like interface.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input parameters.  They are converted to a 2‑D tensor
            and passed through the network.  The output is returned as a
            flattened NumPy array to mimic the quantum version.
        """
        input_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            output = self.forward(input_tensor)
        return output.squeeze().cpu().numpy()


def FCL(n_features: int = 1) -> HybridLayer:
    """
    Factory function kept for backward compatibility with the original
    ``FCL.py`` API.  The returned object behaves like a classical
    fully‑connected layer but uses the richer EstimatorQNN style network.
    """
    return HybridLayer(n_features)


__all__ = ["HybridLayer", "FCL"]
