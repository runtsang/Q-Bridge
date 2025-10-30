"""Classical implementation of a hybrid transformer classifier with optional kernel head."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------- Classical Self‑Attention ----------
class ClassicalSelfAttention(nn.Module):
    """Scaled dot‑product self‑attention implemented with pure PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


# ---------- RBF Kernel ----------
class RBFBasis(nn.Module):
    """Kernel function that returns exp(-γ||x-y||²) for two tensors."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Convenience wrapper that exposes a forward method for two 1‑D tensors."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = RBFBasis(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()


# ---------- Transformer Components ----------
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = ClassicalSelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.dropout(self.attn(x, mask))


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------- Positional Encoding ----------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding as in the original Transformer paper."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------- Unified Classifier ----------
class UnifiedTransformerClassifier(nn.Module):
    """
    A hybrid transformer text classifier that can optionally employ a kernel‑based
    decision layer.  The class mirrors the quantum variant defined in the QML module
    while remaining fully classical.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Dimensionality of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Dropout probability.
    use_kernel : bool, optional
        If ``True``, replace the linear classification head with a learnable
        RBF kernel against a set of class prototypes.
    kernel_gamma : float, optional
        RBF kernel width.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.use_kernel = use_kernel
        if use_kernel:
            # Learnable class prototypes
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
            self.kernel = Kernel(kernel_gamma)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of shape (B, T) containing token indices.
        """
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)  # [B, E]
        x = self.dropout(x)

        if self.use_kernel:
            # Compute kernel similarity to each prototype
            # Shape: [B, C]
            sims = torch.stack(
                [self.kernel(x, proto) for proto in self.prototypes], dim=1
            )
            return sims
        return self.classifier(x)


__all__ = [
    "ClassicalSelfAttention",
    "RBFBasis",
    "Kernel",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "UnifiedTransformerClassifier",
]
