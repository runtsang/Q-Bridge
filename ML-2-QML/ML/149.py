"""Enhanced classical quanvolution filter with attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 patch extraction with a learnable attention mask."""
    def __init__(self) -> None:
        super().__init__()
        # Primary convolution to extract 4-channel patches
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Attention MLP producing a scalar weight per patch
        self.attention = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Flattened attention‑weighted feature vector of shape
            (batch, 4 * 14 * 14).
        """
        patches = self.conv(x)  # (B, 4, 14, 14)
        # Flatten spatial dims
        patches_flat = patches.view(patches.size(0), patches.size(1), -1)  # (B, 4, 196)
        # Compute attention weights per patch
        attn = self.attention(patches_flat.permute(0, 2, 1))  # (B, 196, 1)
        attn = attn.squeeze(-1)  # (B, 196)
        # Apply weights
        weighted = patches_flat.permute(0, 2, 1) * attn.unsqueeze(-1)  # (B, 196, 4)
        # Flatten to feature vector
        return weighted.view(patches.size(0), -1)  # (B, 784)

class QuanvolutionClassifier(nn.Module):
    """Classifier that stacks the quanvolution filter and a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
