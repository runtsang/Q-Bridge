"""Hybrid classical network combining a quanvolution filter and a sampler head.

The implementation expands the original Quanvolution example by
- adding a configurable sampler module that produces a probability
  distribution from the extracted features, and
- exposing a single network class that can be swapped with its quantum
  counterpart for experimentation.

The network is fully Torch‑PyTorch compatible and designed to be
plug‑and‑play in existing pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """
    Classical 2×2 patch encoder mimicking the quantum filter behaviour.
    Applies a 2×2 convolution with stride 2, then reshapes the output
    to a flat feature vector suitable for downstream layers.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        features = self.conv(x)  # (batch, out_channels, H', W')
        return features.view(x.size(0), -1)  # flatten


class ClassicalSampler(nn.Module):
    """
    Lightweight sampler that maps the flattened features to a probability
    distribution over two classes using a small fully‑connected network.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 4, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return F.softmax(logits, dim=-1)


class QuanvolutionSamplerNet(nn.Module):
    """
    End‑to‑end classical network that combines the quanvolution filter
    and sampler head, followed by a classification layer.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    num_classes : int
        Number of target classes for the final linear classifier.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter(in_channels)
        # feature_dim = out_channels * (28//2)^2 = 4 * 14 * 14
        feature_dim = 4 * 14 * 14
        self.sampler = ClassicalSampler(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)          # (batch, feature_dim)
        sampler_out = self.sampler(features)  # (batch, 2)
        logits = self.classifier(features)    # (batch, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, sampler_out


__all__ = ["QuanvolutionSamplerNet", "ClassicalQuanvolutionFilter", "ClassicalSampler"]
