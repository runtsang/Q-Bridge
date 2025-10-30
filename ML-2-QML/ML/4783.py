"""Hybrid regression model combining classical preprocessing, convolution filter, and a sampler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


# --------------------------------------------------------------------------- #
#  Classical preprocessing utilities
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """PyTorch convolutional filter mimicking a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Return a scalar feature for each sample."""
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])  # (batch,)


class SamplerModule(nn.Module):
    """Simple feed‑forward sampler producing a 2‑dimensional probability vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNN() -> SamplerModule:
    """Convenience factory matching the quantum counterpart."""
    return SamplerModule()


# --------------------------------------------------------------------------- #
#  Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding raw features and target values."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Classical hybrid regression combining convolution, sampler, and linear head."""

    def __init__(self, num_features: int, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.sampler = SamplerQNN()
        self.head = nn.Linear(num_features + 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv.run(state_batch).unsqueeze(-1)          # (batch,1)
        # Use first two features as input to sampler
        sampler_feat = self.sampler(state_batch[:, :2])                # (batch,2)
        features = torch.cat([state_batch, conv_feat, sampler_feat], dim=-1)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
