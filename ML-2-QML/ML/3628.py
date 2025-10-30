"""Hybrid kernel method module combining classical RBF kernel and a fully‑connected layer.

The implementation follows the original `QuantumKernelMethod` seed but
replaces the quantum ansatz with a purely classical radial‑basis‑function
kernel.  A lightweight fully‑connected layer is also provided to
demonstrate downstream processing of the kernel embeddings.  The class
can be used directly as a drop‑in replacement for the anchor file and
can later be extended to quantum back‑ends without changing the public
API.

Key design choices
------------------
* **Classical kernel** – A vectorised RBF kernel is implemented using
  NumPy for maximum performance.
* **Fully‑connected layer** – A small `torch.nn.Module` that mirrors the
  behaviour of the quantum fully‑connected layer from the seed.  It
  accepts a list of parameters and returns the mean of a tanh
  transformation.
* **Compatibility** – The public methods (`kernel_matrix` and
  `apply_fcl`) match the signatures of the original seed, so existing
  training scripts can switch to this module without modification.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """Light‑weight replacement for the quantum fully‑connected layer.

    Parameters
    ----------
    n_features : int, default 1
        Number of input features.  The original seed used a single
        feature and we keep the same default for compatibility.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the layer for a list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of parameters that will be fed through the linear
            transform and a tanh non‑linearity.

        Returns
        -------
        np.ndarray
            Mean of the transformed outputs, wrapped in a NumPy array
            to match the quantum implementation.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridKernelMethod:
    """Classical kernel method with optional fully‑connected layer.

    The class exposes a simple RBF kernel and an interface to the
    fully‑connected layer.  It can be used as a drop‑in replacement
    for the original quantum module.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        self.fcl = FullyConnectedLayer()

    # ------------------------------------------------------------------ #
    # Kernel utilities
    # ------------------------------------------------------------------ #
    def _rbf(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the RBF kernel value between two vectors."""
        diff = x - y
        return float(np.exp(-self.gamma * np.sum(diff * diff)))

    def kernel_matrix(
        self,
        a: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Compute the Gram matrix between two datasets.

        Parameters
        ----------
        a, b : Sequence[np.ndarray]
            Datasets represented as iterable of feature vectors.

        Returns
        -------
        np.ndarray
            Gram matrix of shape ``(len(a), len(b))``.
        """
        return np.array(
            [[self._rbf(x, y) for y in b] for x in a],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    # Fully‑connected layer utility
    # ------------------------------------------------------------------ #
    def apply_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Delegate to the underlying fully‑connected layer.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the layer.

        Returns
        -------
        np.ndarray
            Output of the fully‑connected layer.
        """
        return self.fcl.run(thetas)

__all__ = ["HybridKernelMethod", "FullyConnectedLayer"]
