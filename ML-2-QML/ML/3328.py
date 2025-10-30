"""Hybrid classical‑quantum kernel transformer – classical implementation.

This module defines :class:`QuantumKernelTransformer` that uses a
classical RBF kernel and a classical transformer classifier.  It
provides the same public API as the quantum‑centric counterpart so
that existing code can import the class from either the classical
or quantum directory.
"""

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical RBF kernel
# --------------------------------------------------------------------------- #
class ClassicalKernel(nn.Module):
    """Radial‑basis function kernel implemented with PyTorch."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Return the Gram matrix between two lists of torch tensors."""
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Classical transformer blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention implemented in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)

class FeedForward(nn.Module):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# 3. Hybrid class
# --------------------------------------------------------------------------- #
class QuantumKernelTransformer(nn.Module):
    """Hybrid kernel + transformer (classical implementation).

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimensionality.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Dimension of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability.
    kernel_gamma : float, optional
        RBF kernel width.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel = ClassicalKernel(gamma=kernel_gamma)
        self.transformer = TextClassifier(vocab_size, embed_dim, num_heads,
                                          num_blocks, ffn_dim, num_classes,
                                          dropout)

    def compute_kernel_matrix(self,
                              a: Iterable[torch.Tensor],
                              b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sets of input tensors."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer classifier."""
        return self.transformer(x)

__all__ = [
    "ClassicalKernel",
    "kernel_matrix",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
    "QuantumKernelTransformer",
]
