"""
Hybrid classical sampler network with a convolutional feature extractor.
The model mirrors the QNN helper but augments it with a 2‑D convolution
to capture local structure before softmax sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    2‑D convolutional filter inspired by the original Quanvolution example.
    Produces 4‑channel feature maps from a single‑channel input.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class HybridSamplerQuanvolution(nn.Module):
    """
    Classical sampler that first extracts local features via QuanvolutionFilter
    and then produces a probability distribution via a lightweight feed‑forward network.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 4*14*14 features from 28x28 MNIST patches
        self.net = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # two output classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            Tensor of shape (batch, 2) containing class probabilities.
        """
        features = self.qfilter(x)
        logits = self.net(features)
        return F.softmax(logits, dim=-1)


__all__ = ["HybridSamplerQuanvolution"]
