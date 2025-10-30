"""
HybridRegression – Classical convolution + regression architecture.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.
    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of data points to generate.
    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Regression targets of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for synthetic regression data."""
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
# Classical convolutional filter
# --------------------------------------------------------------------------- #

class ConvFilter(nn.Module):
    """A 2‑D convolutional filter with a thresholded sigmoid activation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Single‑channel 2‑D convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).
        Returns
        -------
        torch.Tensor
            Output of shape (batch, 1, H-k+1, W-k+1) after sigmoid thresholding.
        """
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #

class HybridRegression(nn.Module):
    """
    Classical hybrid model that first applies a 2‑D convolutional filter
    to the input features (treated as a square image), then runs a shallow
    multi‑layer perceptron to produce a regression output.
    """
    def __init__(self, num_features: int, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Validate that num_features is a perfect square for 2‑D reshaping
        self.image_dim = int(math.isqrt(num_features))
        assert self.image_dim ** 2 == num_features, (
            "num_features must be a perfect square for the 2‑D convolution."
        )

        self.conv_filter = ConvFilter(kernel_size=kernel_size, threshold=threshold)

        # Output dimensionality after convolution
        conv_out_dim = (self.image_dim - kernel_size + 1) ** 2

        # Simple regression head
        self.net = nn.Sequential(
            nn.Linear(conv_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).
        Returns
        -------
        torch.Tensor
            Regression predictions of shape (batch,).
        """
        bsz = x.shape[0]
        # Reshape to (batch, 1, H, W)
        x = x.view(bsz, 1, self.image_dim, self.image_dim)
        conv_out = self.conv_filter(x)
        # Flatten features for the fully‑connected head
        flat = conv_out.view(bsz, -1)
        return self.net(flat).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
