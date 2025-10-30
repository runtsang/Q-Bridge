"""Classical hybrid model with depthwise separable convolution and noise-aware training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv followed by pointwise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that mimics the original quanvolution structure.
    It replaces the 2×2 convolution with a depthwise separable block and
    optionally injects Gaussian noise during training to study robustness.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 use_depthwise: bool = True, noise_std: float = 0.0) -> None:
        super().__init__()
        self.noise_std = noise_std
        if use_depthwise:
            self.feature_extractor = DepthwiseSeparableConv(in_channels, 4, kernel_size=2, stride=2)
        else:
            self.feature_extractor = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # After 28x28 → 14x14 patches, flattened feature size: 4*14*14
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        features = self.feature_extractor(x)
        features = features.view(x.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def set_noise(self, std: float) -> None:
        """Set the standard deviation for Gaussian noise added to the input."""
        self.noise_std = std

__all__ = ["QuanvolutionHybrid"]
