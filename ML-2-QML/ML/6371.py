"""Hybrid convolutional and regression framework – classical implementation.

The module defines a unified interface `HybridConvRegression` that
provides a classical Conv‑filter, a synthetic regression dataset,
and a small MLP head.  It is fully self‑contained and depends only on
NumPy and PyTorch, making it suitable for rapid classical experiments
or for use as a drop‑in replacement for the quantum layer in legacy code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# Dataset and data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data where labels are a smooth function
    of the sum of features.  The function mirrors the superposition logic
    used in the quantum reference but is expressed purely classically.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical convolution filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """A 2‑D convolution filter that emulates the behaviour of a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the filter to a batch of 2‑D data and return the mean activation."""
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

# --------------------------------------------------------------------------- #
# Regression head
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """Simple feed‑forward regression head."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

# --------------------------------------------------------------------------- #
# Unified interface
# --------------------------------------------------------------------------- #
class HybridConvRegression:
    """
    Unified class exposing both classical convolution and regression capabilities.

    Parameters
    ----------
    mode : str, default="classical"
        Only the classical mode is implemented in this module; the quantum
        mode is deliberately omitted to keep the implementation fully classical.
    """
    def __init__(self, mode: str = "classical", **kwargs) -> None:
        if mode!= "classical":
            raise ValueError("Quantum mode is not supported in the classical implementation.")
        self.conv = ConvFilter(**kwargs)
        self.regressor = QModel(**kwargs)

    def fit(self, dataset: Dataset, epochs: int = 10, lr: float = 1e-3) -> None:
        """Train the regression head on data produced by the convolution filter."""
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.regressor.train()
        for _ in range(epochs):
            for batch in loader:
                features = self.conv(batch["states"].unsqueeze(1))
                preds = self.regressor(features)
                loss = loss_fn(preds, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution then regression to a batch of raw data."""
        with torch.no_grad():
            features = self.conv(data.unsqueeze(1))
            return self.regressor(features)

__all__ = ["HybridConvRegression", "RegressionDataset", "generate_superposition_data"]
