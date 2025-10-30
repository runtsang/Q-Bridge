"""Classical depthwise separable quanvolution filter with a linear head.

This module implements a classical counterpart to the original
quanvolution idea.  It replaces the single 2×2 convolution with a
depth‑wise separable pattern (depth‑wise + point‑wise) to keep the
parameter count low while still extracting 4 feature maps from each
2×2 patch.  A small batch‑norm and dropout regulariser are added
for robustness.  The final linear layer maps the flattened
features to class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Depth‑wise separable quanvolution filter + linear classifier.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    in_channels : int
        Number of input channels (default 1 for MNIST).
    out_channels : int
        Number of output channels per 2×2 patch (default 4).
    kernel_size : int
        Size of the convolution kernel (default 2).
    stride : int
        Stride of the convolution (default 2).
    dropout : float
        Drop‑out probability applied before the linear layer.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Depth‑wise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        # Point‑wise convolution to increase channel count
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        # Linear classifier
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        features = x.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
