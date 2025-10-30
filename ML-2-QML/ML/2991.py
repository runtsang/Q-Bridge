"""Hybrid classical kernel that embeds sequences via a transformer and applies an RBF similarity."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBase(nn.Module):
    """
    A lightweight transformer encoder that can be reused by the kernel.
    It mirrors the architecture of the QTransformerTorch implementation
    but is intentionally kept minimal for speed.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_layers: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                           nhead=num_heads,
                                           dim_feedforward=ffn_dim,
                                           dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).
        mask : torch.Tensor | None
            Binary mask of shape (batch, seq_len) where 0 indicates padding.
        Returns
        -------
        torch.Tensor
            Normalised transformer representation.
        """
        return self.norm(self.encoder(x, src_key_padding_mask=mask))


class TransformerKernel(nn.Module):
    """
    Classical kernel that first projects sequences into a transformer space
    and then applies a radial‑basis function on the resulting embeddings.
    The API matches the original `Kernel` class for drop‑in replacement.
    """
    def __init__(self,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ffn_dim: int = 256,
                 num_layers: int = 2,
                 gamma: float = 1.0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.transformer = TransformerEncoderBase(embed_dim,
                                                  num_heads,
                                                  ffn_dim,
                                                  num_layers,
                                                  dropout)

    def embed(self,
              x: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Embed a batch of sequences using the transformer encoder.
        """
        return self.transformer(x, mask)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of sequences.
        """
        if x.ndim!= 3 or y.ndim!= 3:
            raise ValueError("Input tensors must be 3‑D: (batch, seq_len, embed_dim)")
        x_emb = self.embed(x, mask)
        y_emb = self.embed(y, mask)
        # Collapse the sequence dimension – mean pooling is a simple, effective choice
        x_vec = x_emb.mean(dim=1)
        y_vec = y_emb.mean(dim=1)
        diff = x_vec.unsqueeze(1) - y_vec.unsqueeze(0)
        return torch.exp(-self.gamma * (diff * diff).sum(-1))

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  kernel: TransformerKernel) -> torch.Tensor:
    """
    Vectorised computation of the Gram matrix for two collections of
    sequences.  The function stacks the inputs so that the kernel can
    be evaluated in a single forward pass.
    """
    a_stack = torch.stack(a)  # (n, seq_len, embed_dim)
    b_stack = torch.stack(b)  # (m, seq_len, embed_dim)
    return kernel(a_stack, b_stack)

__all__ = ["TransformerEncoderBase", "TransformerKernel", "kernel_matrix"]
