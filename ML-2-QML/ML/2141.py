"""Enhanced classical quanvolution module with richer feature extraction and uncertainty handling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical quanvolution filter and classifier.
    Uses a 2×2 depthwise convolution followed by a gated linear head.
    Supports a temperature parameter for log_softmax scaling and dropout for MC uncertainty.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 10,
        conv_out_channels: int = 8,
        conv_kernel_size: int = 2,
        conv_stride: int = 2,
        dropout_prob: float = 0.2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=0,
        )
        self.dropout = nn.Dropout(dropout_prob)
        # Compute flattened feature size: (28 - 2) // 2 + 1 = 14
        feature_dim = conv_out_channels * 14 * 14
        self.fc = nn.Linear(feature_dim, out_classes)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        returns log probabilities (batch, out_classes)
        """
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        logits = logits / self.temperature
        return F.log_softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, mc_samples: int = 0) -> torch.Tensor:
        """
        Forward pass with optional MC dropout for uncertainty.
        mc_samples > 0: returns mean logits over mc_samples forward passes.
        """
        if mc_samples <= 0 or not self.training:
            return self.forward(x)
        logits = []
        for _ in range(mc_samples):
            logits.append(self.forward(x).exp())
        return torch.log(torch.stack(logits).mean(0))

__all__ = ["QuanvolutionHybrid"]
