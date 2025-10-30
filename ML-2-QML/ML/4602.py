"""Hybrid quanvolution model – classical implementation.

This module builds upon the original Quanvolution example by fusing a
classical convolutional front‑end with a lightweight random projection
and a fully‑connected head.  The filter can be dropped‑in wherever the
original `QuanvolutionFilter` was used, while the classifier exposes a
log‑softmax output that matches the MNIST 10‑class setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuanvolutionFilter(nn.Module):
    """Simple 2×2 convolutional filter with optional dropout.

    The filter reduces a 28×28 image to a 14×14 grid of 4‑channel
    feature vectors.  The output is flattened to a 4×14×14 vector that
    can be fed into a linear head.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        out_channels: int = 4,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply convolution, optional dropout and flatten."""
        features = self.conv(x)  # (bsz, 4, 14, 14)
        features = self.dropout(features)
        return features.view(x.size(0), -1)  # (bsz, 4*14*14)


class HybridQuanvolutionClassifier(nn.Module):
    """Wraps the filter and a linear classification head."""

    def __init__(
        self,
        num_classes: int = 10,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter(dropout_prob=dropout_prob)
        # 4*14*14 = 784 features from the filter
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)  # (bsz, 784)
        logits = self.linear(features)  # (bsz, num_classes)
        logits = self.batch_norm(logits)
        logits = self.dropout(logits)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionFilter", "HybridQuanvolutionClassifier"]
