"""Hybrid classical model combining CNN, sampler, and custom 2×2 filter.

The module exposes the same API as the original QuantumNAT model
but is fully classical, relying only on PyTorch.  It is designed
for quick experimentation and can be dropped into any PyTorch
training script without modification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset with sinusoidal labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple dataset returning a feature vector and a scalar target."""

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
# Helper modules
# ----------------------------------------------------------------------
class SamplerModule(nn.Module):
    """Lightweight classifier that outputs a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class ConvFilter(nn.Module):
    """2×2 convolutional filter that returns the mean sigmoid activation."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, patch: torch.Tensor) -> float:
        """Compute the mean sigmoid activation for a single patch."""
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class HybridNATModel(nn.Module):
    """Classical hybrid model that mimics the QuantumNAT architecture.

    It consists of:
        * A 2‑layer CNN for feature extraction.
        * A lightweight sampler network that operates on the first two
          flattened features.
        * A custom 2×2 convolutional filter applied over all image patches.
        * A final fully‑connected head producing 4 outputs.
    """

    def __init__(self, input_channels: int = 1, num_features: int = 32) -> None:
        super().__init__()

        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Helper modules
        self.sampler = SamplerModule()
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)

        # Final head
        dummy_input = torch.zeros(1, input_channels, 28, 28)
        dummy_features = self.features(dummy_input).view(1, -1)
        conv_output_dim = 1  # scalar from filter
        sampler_output_dim = 2  # softmax output
        fc_input_dim = dummy_features.size(1) + sampler_output_dim + conv_output_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def _apply_conv_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2×2 filter over all patches and return the mean value."""
        bsz, _, h, w = x.shape
        stride = 1
        kernel = self.conv_filter.kernel_size
        # Extract patches
        patches = (
            x.unfold(2, kernel, stride)
           .unfold(3, kernel, stride)
           .contiguous()
           .view(bsz, -1, kernel, kernel)
        )
        # Compute filter output for each patch
        outputs = torch.stack([self.conv_filter.run(patch) for patch in patches])
        return outputs.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)

        # Sampler works on the first two features
        sampler_input = flat[:, :2]
        sampler_out = self.sampler(sampler_input)

        # Conv filter over all patches
        conv_out = self._apply_conv_filter(x).unsqueeze(-1)

        # Concatenate all signals
        combined = torch.cat([flat, sampler_out, conv_out], dim=1)

        out = self.fc(combined)
        return self.norm(out)


__all__ = ["HybridNATModel", "RegressionDataset", "generate_superposition_data"]
