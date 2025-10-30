"""Enhanced quanvolution model with depthwise separable convolution and multi-task head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv followed by pointwise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class QuanvolutionEnhanced(nn.Module):
    """Hybrid model combining depthwise separable conv and a quantum-inspired patch encoder."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Depthwise separable conv branch
        self.ds_conv = DepthwiseSeparableConv(in_channels, 32, kernel_size=3, stride=1, padding=1)
        # Quantum-inspired patch encoder branch: simple 2x2 patch conv
        self.patch_conv = nn.Conv2d(in_channels, 32, kernel_size=2, stride=2, bias=True)
        # Feature fusion
        self.fusion = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Multi-task heads
        self.classifier = nn.Linear(128 * 7 * 7, num_classes)
        self.regressor = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Depthwise separable conv
        out_ds = self.ds_conv(x)
        # Patch conv
        out_patch = self.patch_conv(x)
        # Concatenate
        out = torch.cat([out_ds, out_patch], dim=1)
        out = F.relu(self.fusion(out))
        out = F.adaptive_avg_pool2d(out, (7, 7))
        out_flat = out.view(out.size(0), -1)
        logits = self.classifier(out_flat)
        aux = self.regressor(out_flat)
        return logits, aux

__all__ = ["QuanvolutionEnhanced", "DepthwiseSeparableConv"]
