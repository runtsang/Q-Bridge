"""Enhanced classical model with hybrid CNN and self‑attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModelEnhanced(nn.Module):
    """CNN backbone with residual connections, followed by a self‑attention
    module and a fully‑connected head that outputs four features."""
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
        )
        # Self‑attention over flattened feature maps
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        features = self.backbone(x)  # shape: (B, 16, H/4, W/4)
        B, C, H, W = features.shape
        # Flatten spatial dims and prepare for attention
        feat_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        # Attention
        attn_out, _ = self.attention(feat_flat, feat_flat, feat_flat)
        # Global average pooling over sequence dimension
        pooled = attn_out.mean(dim=1)  # (B, C)
        # Fully connected head
        out = self.fc(pooled)
        return self.norm(out)

__all__ = ["QFCModelEnhanced"]
