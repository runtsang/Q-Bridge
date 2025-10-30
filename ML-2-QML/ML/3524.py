import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
import torch.utils.data as data


class RBFKernel(nn.Module):
    """Classical RBF kernel implemented in PyTorch."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using RBF kernel."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(data.Dataset):
    """Dataset pairing classical feature vectors with regression targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridKernelRegression(nn.Module):
    """Hybrid kernel ridge regression using a classical RBF kernel."""

    def __init__(self, num_features: int, alpha: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = RBFKernel(gamma)
        self.w = None
        self.X_train = None

    def set_training_data(self, X: torch.Tensor) -> None:
        """Store training features for later prediction."""
        self.X_train = X.detach().clone()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit kernel ridge regression."""
        self.set_training_data(X)
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        K = kernel_matrix(X_np, X_np, self.gamma)
        A = K + self.alpha * np.eye(K.shape[0])
        self.w = np.linalg.solve(A, y_np)

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """Predict using the fitted model."""
        if self.w is None or self.X_train is None:
            raise RuntimeError("Model has not been fitted.")
        X_test_np = X_test.detach().cpu().numpy()
        X_train_np = self.X_train.detach().cpu().numpy()
        K_test = kernel_matrix(X_test_np, X_train_np, self.gamma)
        y_pred = K_test @ self.w
        return torch.tensor(y_pred, dtype=torch.float32)


__all__ = ["RBFKernel", "kernel_matrix", "generate_superposition_data",
           "RegressionDataset", "HybridKernelRegression"]
