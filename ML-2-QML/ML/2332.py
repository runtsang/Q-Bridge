"""Hybrid classical regression model integrating a convolutional filter.

This module defines a classical regression pipeline that
1. Generates superposition‐style data.
2. Applies a 2×2 convolutional filter to each sample.
3. Feeds the scalar filter output into a small neural network.

The design mirrors the original `QuantumRegression.py` but
adds a classical convolutional feature extractor inspired
by the `Conv.py` reference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation – unchanged from the original ML seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset – identical to the original ML seed
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
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
# Classical convolutional filter – borrowed from Conv.py
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2×2 convolution + sigmoid activation used as a feature extractor."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to a 2×2 array and return a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# --------------------------------------------------------------------------- #
# Hybrid classical regression model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """
    Classical regression model that uses a 2×2 convolutional filter
    as a feature extractor followed by a shallow feed‑forward network.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # The filter operates on 2×2 patches; the dataset produces 4‑dim vectors.
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)
        # Network that consumes the scalar filter output.
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convert each sample to a 2×2 array and apply the filter.
        conv_features = torch.tensor(
            [self.conv_filter.run(s.numpy().reshape(2, 2)) for s in state_batch],
            dtype=torch.float32,
        )
        return self.net(conv_features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
