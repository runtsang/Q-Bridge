"""Hybrid classical regression with convolutional feature extraction.

The model combines a 2‑D convolution as a lightweight feature extractor
with a fully‑connected regression head.  It is compatible with the
QuantumRegression.py seed but expands the feature processing pipeline
to demonstrate a more expressive classical baseline."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data that mimics a quantum superposition
    dataset.  The labels are a smooth nonlinear function of the input
    features, ensuring the model must learn a complex mapping.
    """
    x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1  # uniform [-1,1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    """Dataset that reshapes 1‑D features into 2‑D images for convolution."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        # Pad to a square grid
        side = math.isqrt(num_features)
        if side * side!= num_features:
            pad = side * side - num_features
            self.features = torch.nn.functional.pad(self.features, (0, pad))
        self.features = self.features.reshape(-1, side, side)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": self.features[idx], "target": self.labels[idx]}

class ConvFilter(nn.Module):
    """A lightweight 2‑D convolution that emulates a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, height, width)

        Returns
        -------
        torch.Tensor
            Scalar feature map per sample after sigmoid activation.
        """
        x = x.unsqueeze(1)  # add channel dimension
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3]).squeeze(-1)

class QRegressor(nn.Module):
    """
    Classical regression network that first extracts features with a 2‑D
    convolution and then maps them to a scalar output via a small MLP.
    """
    def __init__(self, num_features: int, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        side = math.isqrt(num_features)
        self.feature_dim = side - kernel_size + 1  # due to valid conv
        self.conv = ConvFilter(kernel_size, threshold)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Input of shape (batch, height, width)

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        features = self.conv(states)
        return self.mlp(features).squeeze(-1)

__all__ = ["QRegressor", "RegressionDataset", "generate_superposition_data"]
