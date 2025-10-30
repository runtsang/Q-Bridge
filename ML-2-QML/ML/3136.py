"""Hybrid classical classifier integrating self‑attention and feed‑forward layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class SelfAttentionBlock(nn.Module):
    """Multi‑head self‑attention used inside the classifier."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape  # batch, seq_len, embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B,N,heads,head_dim]
        scores = torch.einsum("bnhd,bmhd->bnm", q, k) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bnm,bmhd->bnhd", attn, v).reshape(B, N, D)
        return self.out_proj(out)

def build_classifier_circuit(num_features: int, depth: int, embed_dim: int = 64) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a hybrid attention‑augmented classifier."""
    layers: List[nn.Module] = []
    in_dim = num_features

    # Linear projection to embed_dim, followed by self‑attention
    layers.append(nn.Linear(in_dim, embed_dim))
    layers.append(SelfAttentionBlock(embed_dim))
    layers.append(nn.Linear(embed_dim, embed_dim))
    layers.append(nn.ReLU())

    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(embed_dim, embed_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

    head = nn.Linear(embed_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # Encoding indices: placeholder for attention parameters
    encoding = [0]
    observables = [1]  # class logits
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
