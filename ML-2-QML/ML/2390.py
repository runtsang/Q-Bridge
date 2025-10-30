"""Combined classical regression with convolution filter.

This module defines a regression model that augments the original
feature vector with a learned convolutional feature extracted by
a 2D Conv filter. The model is a drop‑in replacement for the
original `QModel` but provides an additional feature extraction
stage that improves learning on spatially correlated data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target is a nonlinear
    function of the sum of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RegressionDataset(Dataset):
    """
    PyTorch dataset that yields samples and targets.
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

# ----------------------------------------------------------------------
# Convolution filter (adapted from Conv.py)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    A 2‑D convolution filter that can be applied to a 2‑D patch of
    features.  The filter is trained end‑to‑end with the regression
    network.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution filter.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Filtered output of shape (batch, 1, 1, 1).
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# ----------------------------------------------------------------------
# Regression model
# ----------------------------------------------------------------------
class QModel(nn.Module):
    """
    Classical regression model that augments the raw features with a
    convolutional feature.  The model is compatible with the original
    `QModel` interface but adds a ConvFilter before the fully‑connected
    layers.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # Convolution filter expects a 2x2 patch; enforce num_features==4
        if num_features!= 4:
            raise ValueError("QModel with ConvFilter requires num_features==4")
        self.conv_filter = ConvFilter(kernel_size=2)
        # The regression network now receives 5 inputs: 4 raw + 1 conv
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 4).  The last dimension is reshaped to a
            2x2 patch for the convolution filter.

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (batch,).
        """
        # Reshape to (batch, 1, 2, 2) for ConvFilter
        patch = state_batch.view(state_batch.shape[0], 1, 2, 2)
        conv_out = self.conv_filter(patch)  # (batch, 1, 1, 1)
        conv_out = conv_out.view(state_batch.shape[0], 1)  # (batch, 1)
        # Concatenate raw features and conv output
        features = torch.cat([state_batch, conv_out], dim=1)  # (batch, 5)
        return self.net(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
