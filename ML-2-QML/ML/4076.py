"""Hybrid classical self‑attention with optional regression/classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen042(nn.Module):
    """
    Classical self‑attention module that can act as a regression or classification head.
    """
    def __init__(self, embed_dim: int = 4, depth: int = 2, num_classes: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes

        # linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(0.1)

        # residual transformer‑style blocks
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim))
             for _ in range(depth)]
        )

        # head
        if num_classes is None:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, embed_dim)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)

        for blk in self.blocks:
            out = out + blk(out)

        out = out.mean(dim=1)  # global average pooling

        return self.head(out)

def get_SelfAttentionGen042() -> type:
    """
    Factory that returns the SelfAttentionGen042 class.
    """
    return SelfAttentionGen042

__all__ = ["SelfAttentionGen042", "get_SelfAttentionGen42"]
