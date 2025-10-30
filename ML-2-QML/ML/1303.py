"""Hybrid classical‑quantum transformer for text classification.

The module keeps the original API surface but replaces the
`TextClassifier` with a new class that demonstrates how a
classical transformer can be augmented with quantum‑style
components.  The design is intentionally lightweight so that it can
be run on a CPU or a local simulator; it does not depend on
`torchquantum` but on the standard PyTorch stack.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAttention(nn.Module):
    """Self‑attention that mixes a classical Multi‑head attention with a
    sine‑based quantum style transformation.

    The quantum contribution is weighted by ``quantum_weight`` which
    is a learnable scalar per head.  When ``quantum_weight`` is zero
    the module reduces to a vanilla Multi‑head attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        quantum_scale: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Classical projection layers
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum‑style weight per head
        self.quantum_weight = nn.Parameter(
            torch.full((num_heads, 1), quantum_scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch, seq, _ = x.size()

        # Classical projections
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        classical_out = torch.matmul(attn_weights, v)  # (batch, heads, seq, d_k)

        # Quantum‑style transformation (sine of classical output)
        quantum_out = torch.sin(classical_out)

        # Blend classical and quantum outputs
        weight = torch.sigmoid(self.quantum_weight)  # (heads, 1)
        out = weight * quantum_out + (1 - weight) * classical_out

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class HybridFeedForward(nn.Module):
    """Two‑layer feed‑forward network with optional quantum‑style sine activation."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1,
                 quantum_scale: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Quantum‑style weight
        self.quantum_weight = nn.Parameter(
            torch.full((1,), quantum_scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.linear1(x)
        classical = F.relu(out)
        quantum = torch.sin(out)
        weight = torch.sigmoid(self.quantum_weight)
        out = weight * quantum + (1 - weight) * classical
        return self.linear2(self.dropout(out))


class HybridTransformerBlock(nn.Module):
    """Transformer block that uses hybrid attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, quantum_scale: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = HybridAttention(embed_dim, num_heads, dropout, quantum_scale)
        self.ffn = HybridFeedForward(embed_dim, ffn_dim, dropout, quantum_scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with an optional quantum transform."""
    def __init__(self, embed_dim: int, max_len: int = 5000, quantum_scale: float = 0.0):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Quantum‑style weight
        self.quantum_weight = nn.Parameter(
            torch.full((1,), quantum_scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = x + self.pe[:, : x.size(1)]
        quantum = torch.sin(out)
        weight = torch.sigmoid(self.quantum_weight)
        return weight * quantum + (1 - weight) * out


class QuantumTransformer(nn.Module):
    """Hybrid transformer text classifier.

    The class mirrors the original `TextClassifier` API but replaces
    the transformer blocks with hybrid variants.  All operations are
    implemented with the standard PyTorch stack, so the model can be
    trained on a CPU.
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
        quantum_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = HybridPositionalEncoder(embed_dim, quantum_scale=quantum_scale)
        self.blocks = nn.ModuleList(
            [
                HybridTransformerBlock(
                    embed_dim, num_heads, ffn_dim, dropout, quantum_scale
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        mask = None  # no masking in this simple example
        for block in self.blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "HybridAttention",
    "HybridFeedForward",
    "HybridTransformerBlock",
    "HybridPositionalEncoder",
    "QuantumTransformer",
]
