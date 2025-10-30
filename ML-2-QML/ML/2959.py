"""Hybrid regression module combining classical RBF kernel regression with quantum‑inspired data handling.

The module mirrors the original QuantumRegression seed but adds a lightweight RBF kernel
implementation that can be used by :class:`HybridRegressionModel` for kernel ridge regression.
Both sides share the same data generation and dataset interfaces to enable direct comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition samples for a regression task.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    features, labels : tuple[np.ndarray, np.ndarray]
        ``features`` is a ``(samples, num_features)`` array of real values
        sampled uniformly in ``[-1, 1]``.  ``labels`` are a noisy
        non‑linear transformation of the feature sum to provide a regression
        target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RegressionDataset(Dataset):
    """Torch dataset wrapping the superposition data.

    The dataset yields a dictionary with keys ``states`` (the feature
    vector) and ``target`` (the regression label).  The ``states`` tensor
    is of type ``torch.float32`` to be compatible with both classical
    and quantum encoders.
    """

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Gaussian radial basis function kernel.

    The kernel is defined as
    ``exp(-gamma * ||x - y||^2)`` and is vectorised over batches.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature vectors.  Each element must be a 1‑D tensor.
    gamma : float, default 1.0
        Kernel width.

    Returns
    -------
    np.ndarray
        ``(len(a), len(b))`` Gram matrix in NumPy format.
    """
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #

class HybridRegressionModel(nn.Module):
    """Hybrid regression head that can operate with a classical RBF kernel.

    The model exposes a ``forward`` method that accepts a batch of feature
    vectors and returns the estimated target.  Internally it uses a linear
    head on top of either the raw features or the kernelised feature
    representation, depending on the ``use_kernel`` flag.
    """

    def __init__(
        self,
        num_features: int,
        use_kernel: bool = False,
        gamma: float = 1.0,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.num_features = num_features

        if use_kernel:
            # Kernel ridge regression: we pre‑compute a linear mapping
            # from kernel features to the target.  The mapping is learned
            # by a simple linear layer with no bias.
            self.kernel_size = kernel_size or num_features
            self.kernel_layer = nn.Linear(self.kernel_size, 1, bias=False)
            self.kernel = RBFKernel(gamma)
        else:
            # Plain feed‑forward regression on raw features
            self.net = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.use_kernel:
            # Compute pairwise kernel between batch and a fixed support set.
            # For brevity we use the batch itself as the support set.
            kernel_features = torch.stack(
                [self.kernel(state_batch[i], state_batch) for i in range(state_batch.size(0))]
            )
            # ``kernel_features`` shape: (batch, batch)
            return self.kernel_layer(kernel_features).squeeze(-1)
        else:
            return self.net(state_batch).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "RBFKernel",
    "kernel_matrix",
    "HybridRegressionModel",
]
