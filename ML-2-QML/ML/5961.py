"""Hybrid classical regression model with optional convolutional preprocessing.

This module merges the classical regression seed with the convolutional
filter from Conv.py.  The dataset can optionally apply a 2‑D convolution
filter to each sample, collapsing the feature vector into a single scalar.
The model is a lightweight MLP that accepts either raw features or the
convolution output, demonstrating a simple feature‑engineering pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# 1. Data generation – identical to the classical seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# 2. Convolution filter – drop‑in replacement for the quantum filter
# --------------------------------------------------------------------------- #
def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvFilter(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()

# --------------------------------------------------------------------------- #
# 3. Dataset – supports optional convolution preprocessing
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset that can optionally apply a convolution filter to each sample."""
    def __init__(self, samples: int, num_features: int, conv: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.conv = Conv() if conv else None
        if conv:
            kernel_size = int(np.sqrt(num_features))
            assert kernel_size ** 2 == num_features, "num_features must be a perfect square for conv"
            self.features = np.array(
                [self.conv.run(sample.reshape(kernel_size, kernel_size)) for sample in self.features]
            ).reshape(-1, 1)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# 4. Hybrid regression model – classical MLP with optional conv preprocessing
# --------------------------------------------------------------------------- #
class HybridRegression(nn.Module):
    """Feed‑forward neural network that accepts either raw features or a single
    convolutional scalar per sample.  The architecture is intentionally
    lightweight to keep training fast while demonstrating how a classical
    pre‑processing step can be integrated."""
    def __init__(
        self,
        num_features: int,
        conv: bool = False,
        hidden_sizes: tuple[int, int] = (32, 16),
        device: str = "cpu",
    ):
        super().__init__()
        self.conv = Conv() if conv else None
        input_dim = 1 if conv else num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )
        self.device = device
        self.to(device)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.conv:
            conv_features = []
            for sample in state_batch.cpu().numpy():
                kernel_size = int(np.sqrt(sample.shape[0]))
                conv_features.append(self.conv.run(sample.reshape(kernel_size, kernel_size)))
            state_batch = torch.tensor(conv_features, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.net(state_batch).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "Conv"]
