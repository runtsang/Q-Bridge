"""
HybridKernelRegression – classical implementation.

The class exposes a lightweight RBF kernel and a linear regression
head.  It can be used directly as a torch.nn.Module or wrapped in a
sklearn‑style estimator.  The kernel and dataset utilities are
borrowed from the original seed but renamed and extended to allow
custom gamma values and a simple weighting scheme for future
hybrid experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ClassicalKernalAnsatz(nn.Module):
    """Gaussian RBF kernel implemented in PyTorch."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalKernel(nn.Module):
    """Wrapper that makes the ansatz compatible with ``Kernel`` API."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def generate_classical_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a toy regression dataset with a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset that returns feature vectors and scalar targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridKernelRegression(nn.Module):
    """
    Classical hybrid kernel regression.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature space.
    gamma : float, default=1.0
        Bandwidth of the RBF kernel.
    """

    def __init__(self, num_features: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel = ClassicalKernel(gamma)
        self.regressor = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        The input ``x`` is expected to be a batch of feature vectors.
        The linear head produces a scalar prediction per sample.
        """
        return self.regressor(x).squeeze(-1)


__all__ = [
    "ClassicalKernalAnsatz",
    "ClassicalKernel",
    "kernel_matrix",
    "generate_classical_data",
    "RegressionDataset",
    "HybridKernelRegression",
]
