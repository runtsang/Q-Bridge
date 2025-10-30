"""Hybrid classical self‑attention module that fuses a quanvolutional feature extractor with a learned attention head.

The class exposes a forward interface compatible with the original SelfAttention API
(`rotation_params` and `entangle_params` are accepted for API compatibility but are
currently not used – they can be wired into the linear projections if desired).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Convolutional filter inspired by the original quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridSelfAttention(nn.Module):
    """Hybrid attention module that first extracts local patterns with a quanvolution filter
    and then applies a multi‑head self‑attention mechanism.
    """
    def __init__(self, embed_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.qfilter = QuanvolutionFilter()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: torch.Tensor | None = None,
        entangle_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).
        rotation_params, entangle_params : torch.Tensor, optional
            Placeholders kept for API compatibility with the original SelfAttention
            module. They can be mapped to the linear projections if desired.
        """
        # Extract local features
        qfeat = self.qfilter(x)  # (batch, 4*14*14)
        # Reshape to (batch, seq_len, embed_dim)
        seq_len = qfeat.shape[1] // self.embed_dim
        qfeat = qfeat.view(x.size(0), seq_len, self.embed_dim)

        # Compute attention
        Q = self.query(qfeat)
        K = self.key(qfeat)
        V = self.value(qfeat)

        scores = torch.softmax((Q @ K.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        attn_out = scores @ V
        out = self.out(attn_out)

        return out


__all__ = ["HybridSelfAttention", "QuanvolutionFilter"]
