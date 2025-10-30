"""Hybrid classical estimator combining feed-forward regression and a quantum-inspired convolutional filter.

The module defines `HybridEstimator`, a PyTorch neural network that first extracts 2×2 patches
from a single-channel image via a lightweight convolution, then forwards the flattened
feature vector through a small fully-connected network.  This mirrors the structure of
the original `EstimatorQNN` while incorporating the idea of a quanvolutional filter
from the second reference.
"""

import torch
from torch import nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter producing 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        features = self.conv(x)
        return features.view(x.size(0), -1)  # (batch, 4*14*14)

class HybridEstimator(nn.Module):
    """Hybrid classical estimator with a quanvolution filter and a small regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.regressor = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        return self.regressor(features)

__all__ = ["HybridEstimator"]
