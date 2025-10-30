from __future__ import annotations
import numpy as np
import torch
from torch import nn

class SelfAttention(nn.Module):
    """Classical self‑attention block with learnable projections."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridSelfAttentionEstimator(nn.Module):
    """
    Classical hybrid architecture: self‑attention + small regressor.
    """
    def __init__(self, embed_dim: int = 4, hidden_dims: list[int] = [8, 4]) -> None:
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        return self.regressor(attn_out)

__all__ = ["HybridSelfAttentionEstimator"]
