"""Extended classical quanvolution filter with depth‑aware pooling and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Quanvolution__gen266(nn.Module):
    """
    Classical quanvolutional filter and classifier with adjustable depth and residual connections.
    The filter extracts 2×2 patches, applies a depth‑controlled convolution, optional pooling,
    and a residual skip connection. The classifier head maps the flattened features to logits.
    """
    def __init__(self, depth: int = 1, pool: bool = False) -> None:
        """
        Args:
            depth: Number of times the 2×2 convolution is applied (depth‑controlled).
            pool: If True, apply a 2×2 max‑pool after the convolution.
        """
        super().__init__()
        self.depth = depth
        self.pool = pool

        # Base convolution: 1 input channel -> 4 output channels, 2×2 kernel, stride 2
        self.base_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        # Depth‑controlled additional convolutions
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(4, 4, kernel_size=1, bias=False) for _ in range(depth - 1)
        ])

        # Optional pooling
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

        # Residual connection: 1x1 conv and pooling to match dimensions
        self.residual_conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.residual_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path
        residual = self.residual_conv(x)
        residual = self.residual_pool(residual)

        # Main path
        out = self.base_conv(x)
        for conv in self.depth_convs:
            out = conv(out)
            out = F.relu(out)

        if self.pool:
            out = self.pool_layer(out)

        # Add residual
        out = out + residual
        out = F.relu(out)

        # Flatten and classify
        features = out.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen266"]
