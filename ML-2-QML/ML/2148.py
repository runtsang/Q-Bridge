"""
Hybrid feature extractor with optional quantum kernel, Bayesian head, and dropout for uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionNet(nn.Module):
    """
    Classical implementation of the original Quanvolution network with dropout-based
    Bayesian uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Classical 2x2 convolution acting as the filter
        self.filter = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        # Dropout applied to features before the linear head
        self.dropout = nn.Dropout(dropout)
        # Linear head for classification
        self.linear = nn.Linear(out_channels * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical filter, dropout, and linear head.
        """
        features = self.filter(x)
        features = features.view(x.size(0), -1)
        features = self.dropout(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionNet"]
