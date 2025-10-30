"""Enhanced classical quanvolution network with attention and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical quanvolution network that applies a 2×2 convolution,
    followed by a patch‑wise attention mechanism, dropout and a linear head.
    The architecture keeps the same shape as the original seed but
    introduces learnable weights over image patches.
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = 64,
                 num_classes: int = 10,
                 dropout_rate: float = 0.2) -> None:
        super().__init__()
        # 2×2 convolution with stride 2 -> 14×14 patches, 4 output channels
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        # Attention head: maps 4‑dim patch features to a scalar weight
        self.attention = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layernorm = nn.LayerNorm(4 * 14 * 14)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        features = self.conv(x)                     # (B, 4, 14, 14)
        patches = features.flatten(2).transpose(1, 2)  # (B, 14*14, 4)
        # Compute attention scores for each patch
        attn_scores = self.attention(patches).squeeze(-1)  # (B, 14*14)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, 14*14, 1)
        weighted = patches * attn_weights  # (B, 14*14, 4)
        flat = weighted.reshape(x.size(0), -1)  # (B, 4*14*14)
        flat = self.dropout(flat)
        flat = self.layernorm(flat)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
