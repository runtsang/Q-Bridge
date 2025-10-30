"""Enhanced classical quanvolution filter with learnable depth‑wise separable convolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical depth‑wise separable convolutional filter.

    The filter applies a 2×2 depth‑wise convolution followed by a 1×1
    point‑wise convolution, mirroring the structure of the original
    fixed Conv2d but with fully trainable weights.  A batch
    normalisation and an optional dropout are added to improve
    regularisation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # Depth‑wise 2×2 convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            groups=in_channels,
            bias=True,
        )
        # 1×1 point‑wise convolution to produce the desired output channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Flattened feature vector of shape (B, out_channels * 14 * 14).
        """
        # Depth‑wise 2×2 conv with stride 2 reduces spatial resolution to 14×14
        y = self.depthwise(x)
        # Point‑wise conv mixes channels
        y = self.pointwise(y)
        y = self.bn(y)
        y = self.dropout(y)
        # Flatten to a 1‑D feature vector
        return y.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end classifier using the QuanvolutionFilter.

    The classifier comprises the filter followed by a single linear
    head that maps the extracted features to class logits.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels=4, dropout=0.1)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Log‑softmax probabilities of shape (B, num_classes).
        """
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
