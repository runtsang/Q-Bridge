"""Hybrid kernel regression with classical RBF kernel and ridge regularization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_classical_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset used by both classical and quantum pipelines."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for the generated regression data."""

    def __init__(self, samples: int, num_features: int):
        self.x, self.y = generate_classical_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.x)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.x[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix between X and Y."""
        diff = X[:, None, :] - Y[None, :, :]
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))


class HybridKernelRegression(nn.Module):
    """Classical kernel ridge regression with an RBF kernel."""

    def __init__(self, gamma: float = 1.0, reg: float = 1e-3, device: str = "cpu") -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma)
        self.reg = reg
        self.device = device
        self.alpha = None
        self.X_train = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the model by solving (K + reg * I) alpha = y."""
        X = X.to(self.device)
        y = y.to(self.device)
        K = self.kernel(X, X)
        reg_mat = self.reg * torch.eye(K.shape[0], device=self.device)
        self.alpha = torch.linalg.solve(K + reg_mat, y.unsqueeze(-1)).squeeze(-1)
        self.X_train = X

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict on new data."""
        X = X.to(self.device)
        K_test = self.kernel(X, self.X_train)
        return torch.matmul(K_test, self.alpha)

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Return the R^2 score."""
        preds = self.predict(X)
        return 1 - ((preds - y) ** 2).mean().item()
