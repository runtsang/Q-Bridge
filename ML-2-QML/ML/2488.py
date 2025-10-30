"""Hybrid kernel estimator combining classical RBF kernel and neural network regression."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

# --------------------------------------------------------------------------- #
# Classical kernel primitives – kept for backward compatibility
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Radial‑basis function kernel in torch."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` for single‑sample use."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of torch tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Simple neural‑network regressor
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Fully‑connected network for regression on kernel features."""
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (8, 4)) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

# --------------------------------------------------------------------------- #
# Hybrid estimator – classical side
# --------------------------------------------------------------------------- #
class HybridKernelEstimator:
    """Hybrid RBF‑kernel + neural‑network regressor."""
    def __init__(self, gamma: float = 1.0, hidden_sizes: Sequence[int] = (8, 4),
                 lr: float = 1e-3, epochs: int = 200) -> None:
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.kernel = Kernel(gamma)
        self.model: EstimatorNN | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the neural network on the RBF kernel matrix."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        K = torch.tensor(kernel_matrix(X, X, self.gamma), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.model = EstimatorNN(K.shape[0], self.hidden_sizes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(K)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Predict using the trained model on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        K_test = torch.tensor(kernel_matrix(X, X_train, self.gamma), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(K_test).numpy().flatten()
        return preds

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix",
           "EstimatorNN", "HybridKernelEstimator"]
