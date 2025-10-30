"""Enhanced classical quanvolutional network with depth‑wise separability and residual connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Depth‑wise separable quanvolution filter.
    Each input channel is processed with its own 2x2 kernel, followed by a residual
    connection and dropout for regularisation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Depth‑wise 2x2 convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            groups=in_channels
        )
        # 1x1 convolution for residual path
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W)
        out = self.depthwise(x)
        residual = self.residual(x)
        out = out + residual
        out = self.dropout(out)
        # flatten for linear head
        return out.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network using the depth‑wise quanvolution filter followed by a
    fully‑connected head with a residual block and dropout.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
