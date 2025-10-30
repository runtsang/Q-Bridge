"""Enhanced classical model with residual CNN and self‑attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Classical CNN with residual connections and a self‑attention head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
        )

        self.pool = nn.MaxPool2d(2)

        # Attention module
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(16, embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, 64)
        self.out = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv block
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))

        # Residual connection
        res = self.residual(x1)
        x2 = x2 + res
        x = self.pool(x2)

        # Flatten for attention
        bsz, c, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(bsz, seq_len, c)
        x_proj = self.proj(x_flat)
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        attn_out = attn_out.mean(dim=1)

        # Classifier
        out = F.relu(self.fc(attn_out))
        out = self.out(out)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
