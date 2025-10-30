"""Hybrid classical neural network combining a quanvolution filter, a regression head, and optional shot noise.

The implementation merges concepts from:
- The original quanvolution filter (2×2 conv + flatten)
- The EstimatorQNN regression network
- The FastEstimator noise wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter that downsamples the image."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class EstimatorQNN(nn.Module):
    """Fully‑connected regression head inspired by the EstimatorQNN example."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQuanvolutionNet(nn.Module):
    """
    Classical hybrid network:
    quanvolution filter → flatten → EstimatorQNN head.
    Optionally adds Gaussian shot noise to mimic quantum measurement statistics.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        use_noise: bool = False,
        shots: int | None = None,
    ):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        self.head = EstimatorQNN(self.qfilter.conv.out_channels * 14 * 14)
        self.use_noise = use_noise
        self.shots = shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.head(features)
        if self.use_noise and self.shots is not None:
            rng = np.random.default_rng()
            noise = rng.normal(0, 1 / np.sqrt(self.shots), size=logits.shape)
            logits = logits + torch.tensor(noise, dtype=logits.dtype, device=logits.device)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "EstimatorQNN", "HybridQuanvolutionNet"]
