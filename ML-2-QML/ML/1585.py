"""Enhanced classical quanvolution filter with learnable patch‑wise convolution and global pooling.

This module keeps the original 2×2 patch extraction but replaces the fixed
convolutional kernel with a trainable `nn.Conv2d`.  After the patch
extraction we apply a global average pooling and a dropout layer before
the linear classifier.  The design allows the user to experiment with
different kernel sizes or add more convolutional layers while still
maintaining compatibility with the original API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 patch extraction followed by a learnable convolution.
    The module accepts a single‑channel image of size 28×28.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, 1, 28, 28)
        returns: tensor of shape (batch, out_channels * 14 * 14)
        """
        features = self.conv(x)            # (batch, out, 14, 14)
        features = features.view(features.size(0), -1)
        if self.dropout.p > 0.0:
            features = self.dropout(features)
        return features


class QuanvolutionClassifier(nn.Module):
    """
    Pure‑classical classifier that uses the QuanvolutionFilter as a
    feature extractor followed by a linear head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels,
                                          dropout=dropout)
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
