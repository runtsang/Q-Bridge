"""
Hybrid classical regression model combining a convolutional feature extractor and a fully‑connected head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target depends on a nonlinear
    transformation of the input. The data are also reshaped to a 2‑D grid
    suitable for a convolutional filter.
    """
    # Uniformly sample in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Nonlinear target
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Reshape to a square grid (kernel_size x kernel_size)
    # Find the smallest integer > sqrt(num_features)
    k = int(np.ceil(np.sqrt(num_features)))
    padded = np.pad(x, ((0, 0), (0, k * k - num_features)), mode="constant")
    x_grid = padded.reshape(samples, k, k)
    return x_grid, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset yielding 2‑D feature grids and scalar targets.
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


class ConvFilter(nn.Module):
    """
    A lightweight 2‑D convolutional filter that mimics the behaviour of the
    original quanvolution layer but runs fully classically.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the convolution and a sigmoid activation.
        """
        # data shape: (batch, height, width)
        x = data.unsqueeze(1)  # add channel dimension
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Global average pooling to scalar per example
        return activations.mean(dim=[2, 3])


class HybridRegression(nn.Module):
    """
    Classical regression model that optionally applies a ConvFilter before
    a small MLP.  The interface is identical to the original QModel.
    """

    def __init__(self, num_features: int, use_conv: bool = False, kernel_size: int = 2):
        super().__init__()
        self.use_conv = use_conv
        self.kernel_size = kernel_size
        self.num_features = num_features

        if self.use_conv:
            # Ensure the grid size matches the kernel
            self.conv = ConvFilter(kernel_size=kernel_size)
            # After conv we have a scalar per example
            self.net = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            # Plain MLP
            self.net = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through either the conv+MLP or plain MLP.
        """
        if self.use_conv:
            # Flatten the 2‑D grid into a 1‑D vector for the linear head
            conv_out = self.conv(state_batch)  # shape: (batch, 1)
            return self.net(conv_out).squeeze(-1)
        else:
            return self.net(state_batch.view(state_batch.size(0), -1)).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
