"""
Enhanced classical quanvolution filter with a trainable 2×2 convolution.
All patches share the same weights, enabling end‑to‑end learning while keeping
the computational cost identical to the baseline implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Trainable 2×2 convolution that emulates patch extraction.
    Padding is zero to match the original 2×2 patch size.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """
    Classic classifier that stacks the trainable quanvolution filter
    with a fully‑connected head.  The output features are 4×14×14 for 28×28 images.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, hidden_dim)
        self.linear = nn.Linear(hidden_dim * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
