"""Hybrid classical sampler with quantum‑inspired convolution and regression head.

This module fuses the simple SamplerQNN architecture with a quantum‑style
convolutional feature extractor (inspired by Quanvolution) and a regression
head (similar to EstimatorQNN).  The network can be used as a drop‑in
replacement for a classical sampler while still exposing a quantum‑like
feature extraction pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler network that combines:
        * a classical 2‑D convolutional filter (mimicking a quanvolutional layer)
        * a softmax sampler (2 → 4 → 2)
        * a regression head (2 → 8 → 4 → 1)
    The forward pass returns both the sampled probability distribution
    and the regression output.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # Convolutional feature extractor (mimics a quanvolutional filter)
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=True)

        # Sampler head
        feature_dim = 4 * 14 * 14  # 28×28 image → 14×14 patches after stride 2
        self.sampler = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_classes, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            probs: Softmax probabilities of shape (batch, num_classes)
            regression: Scalar regression output of shape (batch, 1)
        """
        # Convolutional features
        feat = self.conv(x)                     # (batch, 4, 14, 14)
        flat = feat.view(feat.size(0), -1)      # (batch, 4*14*14)

        # Sampler probabilities
        logits = self.sampler(flat)             # (batch, 2)
        probs = F.softmax(logits, dim=-1)       # (batch, 2)

        # Regression output
        regression = self.regressor(probs)      # (batch, 1)

        return probs, regression

__all__ = ["HybridSamplerQNN"]
