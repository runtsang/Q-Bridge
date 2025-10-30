"""Hybrid classical regression model combining convolutional filtering with a linear head.

The model processes 2‑D superposition data, applies a lightweight
convolutional filter, flattens the resulting feature map and
produces a scalar output.  It is a drop‑in replacement for the
original ``QModel`` but uses only NumPy/PyTorch primitives.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset.

    Parameters
    ----------
    num_features : int
        Number of features per sample.  Must be a perfect square so that
        the features can be reshaped into a 2‑D patch.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``x`` of shape ``(samples, H, W)`` and scalar labels ``y``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    H = W = int(np.sqrt(num_features))
    x = x.reshape(samples, H, W)
    angles = x.sum(axis=(1, 2))
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple PyTorch ``Dataset`` for the superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ConvFilter(nn.Module):
    """Lightweight 2‑D convolutional filter inspired by the original
    ``Conv`` helper.  The filter produces a single‑channel feature map
    that is later flattened and fed to a linear head.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolution, a sigmoid activation and return the
        resulting feature map.

        Parameters
        ----------
        data : torch.Tensor
            Input of shape ``(batch, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map of shape ``(batch, 1, H-1, W-1)``.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


class HybridRegressionModel(nn.Module):
    """
    Classical regression model that combines a convolutional filter
    with a linear head.  It mirrors the structure of the original
    quantum model but uses only CPU‑side PyTorch operations.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        H = int(np.sqrt(num_features))
        self.num_patches = (H - 1) * (H - 1)
        self.filter = ConvFilter(kernel_size=2)
        self.head = nn.Linear(self.num_patches, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape ``(batch, H, W)`` where ``H*W == num_features``.

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape ``(batch,)``.
        """
        x = state_batch.unsqueeze(1)  # add channel dimension
        features = self.filter(x)  # (batch, 1, H-1, W-1)
        features = features.view(features.size(0), -1)  # flatten
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
