"""Enhanced classical quanvolutional architecture with depthwise separable convs, residuals and a multi‑scale patch extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.pointwise(self.depthwise(x)))


class ResidualBlock(nn.Module):
    """Simple residual block built from depthwise separable convs."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))


class MultiScalePatchExtractor(nn.Module):
    """Extracts flattened 2×2 and 4×4 patches from an image."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = x.shape
        patches = []
        for scale in (2, 4):
            stride = scale
            unfolded = F.unfold(x, kernel_size=scale, stride=stride)
            unfolded = unfolded.transpose(1, 2)  # shape: [B, num_patches, scale*scale*c]
            patches.append(unfolded)
        return torch.cat(patches, dim=1)  # [B, total_patches, patch_dim]


class QuanvolutionFilter(nn.Module):
    """Classical depthwise‑separable quanvolution filter with residual connections."""
    def __init__(self):
        super().__init__()
        self.patch_extractor = MultiScalePatchExtractor()
        # Conv to reduce dimensionality before residual
        self.pre_conv = DepthwiseSeparableConv2d(1, 4, kernel_size=2, stride=2)
        self.residual = ResidualBlock(4)
        # Compute feature size: 196 patches from 2×2 + 49 patches from 4×4
        # Each patch has 4 or 16 elements respectively
        self.feature_dim = 196 * 4 + 49 * 16
        self.fc = nn.Linear(self.feature_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch extraction
        patches = self.patch_extractor(x)  # [B, total_patches, patch_dim]
        # Reduce dimensionality with a lightweight conv
        patches = patches.view(x.size(0), 1, -1, patches.size(2))
        patches = self.pre_conv(patches)
        # Residual processing
        patches = self.residual(patches)
        # Flatten and linear projection
        flat = patches.view(patches.size(0), -1)
        return self.fc(flat)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the enhanced quanvolution filter."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
