"""Enhanced classical regression model with residual connections, dropout, and batch normalization."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy superposition target data.

    The function creates states of the form
    cos(theta) |0...0> + e^{i phi} sin(theta) |1...1>
    and maps them to a real‑valued target via
    sin(2*theta) * cos(phi). A small Gaussian noise is added
    to the target to make the regression task more realistic.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(2 * angles) * np.cos(angles)  # deterministic signal
    noise = np.random.normal(scale=0.05, size=y.shape)
    return x, (y + noise).astype(np.float32)

class RegressionDataset(Dataset):
    """Simple torch Dataset for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """A lightweight feed‑forward network with optional residuals and dropout.

    The network is intentionally simple yet expressive enough for the
    noisy superposition regression task. It includes:
      * Two hidden layers with 64 units each.
      * Batch‑normalisation after each hidden layer.
      * Dropout (p=0.2) to reduce over‑fitting.
      * Optional residual connection between the input and the second hidden layer.
    """
    def __init__(self, num_features: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.head = nn.Linear(64, 1)
        if use_residual:
            # Project input to match hidden dimension for addition
            self.res_proj = nn.Linear(num_features, 64)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(state_batch)
        if self.use_residual:
            x = x + self.res_proj(state_batch)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
