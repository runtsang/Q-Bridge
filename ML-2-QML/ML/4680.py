"""Hybrid kernel regression framework.

This module implements a classical RBF kernel, a quantum kernel interface, and
a simple kernel‑ridge regression model that can mix both kernels.  It also
includes a lightweight sampler network (SamplerQNN) that can be used as a
feature extractor before kernel evaluation.

The design follows the seed in QuantumKernelMethod.py but extends it with
quantum‑classical coupling, dataset utilities, and a regression model.
"""

from __future__ import annotations

from typing import Callable, Sequence, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Classical RBF kernel implementation (seed 1)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel.

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel width.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x-y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# SamplerQNN (seed 2)
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    """Simple neural network that mimics the quantum SamplerQNN.

    It produces a probability distribution over two outputs, which can be
    interpreted as a classical feature map before kernel evaluation.
    """
    class _SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return _SamplerModule()

# --------------------------------------------------------------------------- #
# Dataset utilities (seed 3)
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    The target is a smooth non‑linear function of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset providing (state, target) pairs for regression."""

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid kernel computation
# --------------------------------------------------------------------------- #
def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    q_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """Compute the Gram matrix between ``a`` and ``b``.

    Parameters
    ----------
    a, b : sequence of 1‑D tensors
        Input samples.
    gamma : float
        Width of the classical RBF kernel.
    q_kernel : callable, optional
        Quantum kernel function that accepts two tensors and returns a scalar.
        If provided, the returned matrix is the sum of the classical and quantum
        kernels.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    rbf = RBFKernel(gamma)
    classical = np.array(
        [[rbf(x, y).item() for y in b] for x in a]
    )
    if q_kernel is None:
        return classical
    quantum = np.array(
        [[q_kernel(x, y).item() for y in b] for x in a]
    )
    return classical + quantum

# --------------------------------------------------------------------------- #
# Kernel Ridge Regression (simple linear algebra solver)
# --------------------------------------------------------------------------- #
class KernelRidgeRegression(nn.Module):
    """Kernel Ridge Regression using a hybrid kernel."""

    def __init__(self, gamma: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X: torch.Tensor, y: torch.Tensor, q_kernel: Optional[Callable] = None) -> None:
        """Fit the model to data ``X`` and targets ``y``."""
        K = torch.tensor(
            kernel_matrix(X, X, self.gamma, q_kernel),
            dtype=torch.float32,
            device=X.device,
        )
        # Closed form solution: (K + alpha * I)^{-1} y
        self.coef_ = torch.linalg.solve(K + self.alpha * torch.eye(K.shape[0], device=K.device), y)

    def predict(self, X: torch.Tensor, q_kernel: Optional[Callable] = None) -> torch.Tensor:
        """Predict using the fitted kernel."""
        K_test = torch.tensor(
            kernel_matrix(X, self.train_X, self.gamma, q_kernel),
            dtype=torch.float32,
            device=X.device,
        )
        return torch.matmul(K_test, self.coef_)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "RBFKernel",
    "SamplerQNN",
    "generate_superposition_data",
    "RegressionDataset",
    "kernel_matrix",
    "KernelRidgeRegression",
]
