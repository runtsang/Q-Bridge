"""Enhanced classical model with residual block and self‑attention for Quantum‑NAT.

This module extends the original QFCModel by:
1. Adding a residual block after the convolutional backbone.
2. Applying a multi‑head self‑attention over the flattened feature map.
3. Using a learnable projection head that maps the attended representation to 4 output features.
4. Normalising the final output with BatchNorm1d.

The architecture remains fully classical and can be trained with standard
optimisers from PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Classical convolutional‑attention model for Quantum‑NAT."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Residual block
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
        )
        self.residual = nn.Conv2d(16, 16, kernel_size=1)
        # Self‑attention
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        feat = self.backbone(x)
        # Residual connection
        feat = feat + self.residual(feat)
        # Flatten spatial dims
        bsz, c, h, w = feat.shape
        feat_flat = feat.view(bsz, c, -1).transpose(1, 2)  # (B, N, C)
        # Self‑attention
        attn_out, _ = self.attn(feat_flat, feat_flat, feat_flat)
        # Merge attention output
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, -1)
        # Projection
        out = self.proj(attn_out)
        return self.norm(out)

__all__ = ["QFCModel"]
