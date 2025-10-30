"""Enhanced classical regression framework with advanced preprocessing and training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data resembling a quantum superposition distribution.
    Each sample is a vector of continuous features; the target is a
    non‑linear combination of sines and cosines of a random projection
    to emulate interference patterns.
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    proj = X @ np.random.randn(num_features).astype(np.float32)
    y = np.sin(proj) + 0.1 * np.cos(2 * proj) + 0.05 * np.random.randn(samples)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper returning features and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

class QuantumRegressionModel(nn.Module):
    """Feed‑forward network tuned for regression on quantum‑style data."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)

def train(model: nn.Module,
          dataset: Dataset,
          epochs: int = 200,
          lr: float = 1e-3,
          batch_size: int = 64,
          device: str | torch.device = "cpu",
          verbose: bool = True) -> list[float]:
    """
    Simple training loop for regression.
    Returns a list of epoch‑level MSE losses.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/{epochs:03d} | MSE: {epoch_loss:.4f}")
    return losses

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data", "train"]
