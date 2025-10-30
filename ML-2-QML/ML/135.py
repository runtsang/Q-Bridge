"""Enhanced classical quanvolution filter with optional depthwise separable convolution.

This module implements a convolutional filter inspired by the original
`QuanvolutionFilter` but with additional features:
  * configurable patch size and number of output channels
  * optional depthwise separable convolution for efficiency
  * batch normalization and dropout for regularisation
  * a lightweight classifier head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen148(nn.Module):
    """
    Classical quanvolutional network.

    Parameters
    ----------
    patch_size : int, default 2
        Size of the square patch to convolve over.
    out_channels : int, default 4
        Number of output channels of the convolution.
    depthwise : bool, default False
        If True use a depthwise separable convolution.
    dropout_prob : float, default 0.0
        Dropout probability applied after the convolution.
    """
    def __init__(
        self,
        patch_size: int = 2,
        out_channels: int = 4,
        depthwise: bool = False,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        stride = patch_size
        if depthwise:
            # depthwise separable: conv per input channel then pointwise
            self.depthwise = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=patch_size,
                stride=stride,
                groups=1,
                bias=False,
            )
            self.pointwise = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        else:
            self.depthwise = None
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=stride,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        # compute flattened feature size: out_channels * (28//patch_size)**2
        self.feature_dim = out_channels * (28 // patch_size) ** 2
        self.classifier = nn.Linear(self.feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28)
        if self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        # flatten
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened feature vector before the classifier."""
        if self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)


__all__ = ["Quanvolution__gen148"]
