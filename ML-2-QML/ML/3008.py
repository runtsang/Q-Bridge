"""Hybrid classical model combining a quanvolution filter with a feed‑forward estimator.

The model first extracts 2×2 patches via a 2‑D convolution, then passes the flattened
feature map through a small neural network that mirrors the EstimatorQNN architecture.
This design preserves the locality of the original quanvolution while adding a
classical regression head inspired by the EstimatorQNN seed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that emulates the 2×2 patch extraction of the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class EstimatorNN(nn.Module):
    """Feed‑forward regressor that mirrors the EstimatorQNN architecture."""
    def __init__(self, input_dim: int, output_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class QuanvolutionHybrid(nn.Module):
    """Hybrid classical model: quanvolution filter + estimator head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        # filter output dimension: 4 * 14 * 14 = 784 for 28×28 MNIST images
        self.estimator = EstimatorNN(input_dim=4 * 14 * 14, output_dim=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.estimator(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "EstimatorNN", "QuanvolutionHybrid"]
