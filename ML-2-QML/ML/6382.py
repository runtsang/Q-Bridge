"""Enhanced classical Quanvolution network with trainable kernels, dropout, and L2 regularization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter that mimics the original 2×2 stride‑2 conv.
    The weights are trainable and can be regularized via weight decay.
    Dropout is applied to the feature map to reduce overfitting.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, H, W)
        Returns:
            Tensor of shape (batch, out_channels * (H//stride) * (W//stride))
        """
        feat = self.conv(x)
        feat = self.dropout(feat)
        return feat.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end classifier that uses the QuanvolutionFilter followed by a
    fully‑connected head. The linear layer is initialized with Xavier normal
    and includes L2 regularization via weight decay during optimizer configuration.
    """

    def __init__(
        self,
        num_classes: int = 10,
        filter_out_channels: int = 4,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(
            out_channels=filter_out_channels, dropout_prob=dropout_prob
        )
        # 28×28 input → 14×14 patches → 14×14×filter_out_channels features
        self.linear = nn.Linear(filter_out_channels * 14 * 14, num_classes, bias=True)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            Log‑softmax logits of shape (batch, num_classes)
        """
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
