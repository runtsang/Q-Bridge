"""Hybrid convolutional filter with optional residual and feature extractor.

This module implements ConvFilter that works purely classically.
"""

import torch
from torch import nn
import numpy as np
from typing import Optional

class ConvFilter(nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        residual: bool = False,
        feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.residual = residual
        self.feature_extractor = feature_extractor

        # learnable 2‑D kernel
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
            stride=1,
            padding=0,
        )
        # initialise weights
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, 1, H, W)."""
        out = self.conv(data)
        if self.residual:
            # add residual connection
            out = out + data
        # binarise with threshold
        out = torch.sigmoid(out - self.threshold)
        if self.feature_extractor is not None:
            out = self.feature_extractor(out)
        return out

    def run(self, data: np.ndarray) -> float:
        """Convenience method that accepts a 2‑D numpy array and returns a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = self.forward(tensor)
        return out.mean().item()
