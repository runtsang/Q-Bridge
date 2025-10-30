"""Hybrid transformer with quantum‑inspired linear layers and optional quantum patch embeddings.

The module mirrors the original QTransformerTorch but adds:
- RandomLinear: a fixed orthogonal matrix used as a quantum‑inspired linear transform.
- QuantumMultiHeadAttention and QuantumFeedForward: optional usage of RandomLinear for projections.
- QuantumPatchEmbedding: optional quantum‑inspired patch embedding applied to token vectors.
- Fraud detection style clipping: all random matrices are clipped to [-5,5] upon initialization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomLinear(nn.Module):
    """Fixed random orthogonal linear transform with optional clipping."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, clip: bool = True) -> None:
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(out_features, in_features))
        if clip:
            q = q.clamp(-5.0, 5.0)
        self.weight = nn.Parameter(q.t(), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input @ self.weight
        if self.bias is not None:
            output += self.bias
        return output


class QuantumMultiHeadAttention(nn.Module):
    """Multi‑head attention that optionally uses quantum‑inspired linear projections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, quantum: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.quantum = quantum
        projection_cls = RandomLinear if quantum else nn.Linear
        self.k_linear = projection_cls(embed_dim, embed_dim, bias=False)
        self.q_linear = projection_cls(embed_dim, embed_dim, bias=False)
        self.v_linear = projection_cls(embed_dim, embed_dim, bias=False)
        self.combine_heads = projection_cls(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


class QuantumFeedForward(nn.Module):
    """Feed‑forward block that optionally uses quantum‑inspired linear transforms."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, quantum: bool = False) -> None:
        super().__init__()
        projection_cls = RandomLinear if quantum else nn.Linear
        self.linear1 = projection_cls(embed_dim, ffn_dim, bias=True)
        self.linear2 = projection_cls(ffn_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 quantum_attention: bool = False,
                 quantum_ffn: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout, quantum=quantum_attention)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout, quantum=quantum_ffn)
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
        return x + self.pe[:, :x.size(1)]


class QuantumPatchEmbedding(nn.Module):
    """Optional quantum‑inspired patch embedding applied to token vectors."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.random_proj = RandomLinear(embed_dim, embed_dim, bias=False, clip=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_proj(x)


class QTransformerGen320(nn.Module):
    """Hybrid transformer classifier with optional quantum‑inspired components."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 quantum_attention: bool = False,
                 quantum_ffn: bool = False,
                 quantum_patch: bool = False) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.quantum_patch = quantum_patch
        if quantum_patch:
            self.patch_embed = QuantumPatchEmbedding(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout,
                               quantum_attention=quantum_attention,
                               quantum_ffn=quantum_ffn)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        if self.quantum_patch:
            tokens = self.patch_embed(tokens)
        x = self.positional(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "RandomLinear",
    "QuantumMultiHeadAttention",
    "QuantumFeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "QuantumPatchEmbedding",
    "QTransformerGen320",
]
