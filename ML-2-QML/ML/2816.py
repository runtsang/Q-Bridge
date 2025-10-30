"""Hybrid transformer classifier with optional classical Self‑Attention blocks.

This module contains a pure‑classical implementation that mimics the API of
the original QTransformerTorch but adds an alternative Self‑Attention block
derived from the SelfAttention reference.  The class can be instantiated
with ``use_self_attention=True`` to replace the standard multi‑head
attention with a self‑attention block that uses linear projections
parameterised by rotation and entanglement matrices.

The implementation is fully compatible with PyTorch 1.10+ and
requires only ``torch`` and ``numpy``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block that mimics the interface of the reference
    SelfAttention implementation.  The rotation and entanglement matrices
    are learned parameters of the block and default to random orthogonal
    matrices if not supplied.
    """

    def __init__(
        self,
        embed_dim: int,
        rotation_params: Optional[np.ndarray] = None,
        entangle_params: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # initialise parameters
        if rotation_params is None:
            rotation_params = np.linalg.qr(np.random.randn(embed_dim, embed_dim))[0]
        if entangle_params is None:
            entangle_params = np.linalg.qr(np.random.randn(embed_dim, embed_dim))[0]
        self.register_buffer("rotation_params", torch.from_numpy(rotation_params).float())
        self.register_buffer("entangle_params", torch.from_numpy(entangle_params).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Output of the self‑attention block with the same shape as ``x``.
        """
        # Linear projections
        query = torch.matmul(x, self.rotation_params)
        key = torch.matmul(x, self.entangle_params)
        # Attention scores
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim),
            dim=-1,
        )
        # Weighted sum
        return torch.matmul(scores, x)


class FeedForwardClassical(nn.Module):
    """Standard two‑layer feed‑forward network used in transformer blocks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding as in “Attention Is All You Need”."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlockClassical(nn.Module):
    """Standard transformer block with multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class SelfAttentionTransformerBlock(nn.Module):
    """Transformer block that replaces multi‑head attention with the
    classical Self‑Attention implementation.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        rotation_params: Optional[np.ndarray] = None,
        entangle_params: Optional[np.ndarray] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = ClassicalSelfAttention(
            embed_dim, rotation_params, entangle_params
        )
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformerClassifier(nn.Module):
    """Hybrid transformer‑based text classifier that can operate in three modes:

    * ``use_self_attention=False`` – standard multi‑head transformer.
    * ``use_self_attention=True`` – replace attention with classical
      self‑attention blocks.
    * ``quantum_override=True`` – use the quantum implementation provided
      in :py:mod:`qml_code`.  This flag is only recognised when the module
      is imported from the quantum file; it is ignored in the classical
      implementation to keep the API identical.

    The class is fully compatible with the original QTransformerTorch
    classifier and can be dropped‑in as a drop‑in replacement.
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
        use_self_attention: bool = False,
        rotation_params_list: Optional[Iterable[np.ndarray]] = None,
        entangle_params_list: Optional[Iterable[np.ndarray]] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = (
            SelfAttentionTransformerBlock
            if use_self_attention
            else TransformerBlockClassical
        )
        if use_self_attention:
            # If user supplied parameter lists, recycle them; otherwise
            # generate random ones on the fly for each block.
            if rotation_params_list is None:
                rotation_params_list = [
                    np.linalg.qr(np.random.randn(embed_dim, embed_dim))[0]
                    for _ in range(num_blocks)
                ]
            if entangle_params_list is None:
                entangle_params_list = [
                    np.linalg.qr(np.random.randn(embed_dim, embed_dim))[0]
                    for _ in range(num_blocks)
                ]
            blocks = [
                block_cls(
                    embed_dim,
                    ffn_dim,
                    rotation_params=rotation_params_list[i],
                    entangle_params=entangle_params_list[i],
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        else:
            blocks = [
                block_cls(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of token indices with shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)`` or ``(batch, 1)`` for binary.
        """
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "ClassicalSelfAttention",
    "FeedForwardClassical",
    "PositionalEncoder",
    "TransformerBlockClassical",
    "SelfAttentionTransformerBlock",
    "HybridTransformerClassifier",
]
