"""Hybrid kernel regression framework with classical RBF kernel."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data using a sinusoidal function of random angles.
    The data mimics the structure of a superposition state but remains purely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper for the synthetic regression data.
    Each sample is a dictionary containing the feature vector and the target value.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class KernalAnsatz(nn.Module):
    """Classical RBF kernel implementation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that exposes a callable kernel matrix."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two sets of samples using an RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class HybridKernelRegression(nn.Module):
    """
    Classical kernel ridge regression using an RBF kernel.
    The model learns a linear combination of kernel evaluations.
    """
    def __init__(self, num_features: int, gamma: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_features = num_features
        self.train_X = None
        self.train_y = None
        self.w = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the model by solving the regularized least-squares problem
        (K + alpha*I) w = y, where K is the kernel matrix.
        """
        self.train_X = X.detach().clone()
        self.train_y = y.detach().clone()
        K = kernel_matrix(X, X, self.gamma)
        K += self.alpha * np.eye(K.shape[0])
        self.w = np.linalg.solve(K, self.train_y.numpy())

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict targets for new samples by evaluating the kernel against
        the training data and applying the learned weights.
        """
        if self.w is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = kernel_matrix(X, self.train_X, self.gamma)
        preds = K @ self.w
        return torch.tensor(preds, dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "RegressionDataset",
    "HybridKernelRegression",
]
