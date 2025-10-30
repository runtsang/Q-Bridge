"""Hybrid transformer with optional quantum layers – classical baseline implementation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical building blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared base for attention layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B,N,E) → (B,N,H,d_k)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        return self.dropout(scores) @ v

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Must be overridden by subclasses."""
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Purely classical multi‑head attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        qk = self.separate_heads(q)
        kq = self.separate_heads(k)
        vq = self.separate_heads(v)
        out = self.attention(qk, kq, vq)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out_proj(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑augmented attention that routes a linear projection through a small variational circuit."""

    class _QLayer(nn.Module):
        """Simple two‑qubit variational ansatz implemented as a learnable rotation."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.wire_vars = nn.Parameter(torch.randn(n_wires, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Fake quantum layer: apply a small rotation per wire
            batch, seq, dim = x.shape
            x = x.reshape(batch * seq, dim)
            rotations = torch.einsum("bi,ij->bj", x, self.wire_vars)
            return rotations.reshape(batch, seq, dim)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
        n_wires: int = 4,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.q_layer = self._QLayer(n_wires)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_layer(x)
        k = self.q_layer(x)
        v = self.q_layer(x)
        qk = self.separate_heads(q)
        kq = self.separate_heads(k)
        vq = self.separate_heads(v)
        out = self.attention(qk, kq, vq)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
#  Feed‑forward blocks
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for all FFNs."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP with ReLU activation."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a simple variational module."""

    class _QLayer(nn.Module):
        """Learnable rotation applied per token."""

        def __init__(self, n_qubits: int = 4):
            super().__init__()
            self.n_qubits = n_qubits
            self.params = nn.Parameter(torch.randn(n_qubits, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Apply a small rotation per qubit and return the transformed vector
            batch, seq, dim = x.shape
            x = x.reshape(batch * seq, dim)
            rot = torch.einsum("bi,ij->bj", x, self.params)
            return rot.reshape(batch, seq, dim)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply quantum layer per token
        q_out = self.q_layer(x)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockHybrid(nn.Module):
    """Hybrid block that can run either classical or quantum sub‑modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_wires: int = 4,
        n_qubits_ffn: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_wires)
            if use_quantum_attention
            else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if use_quantum_ffn
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
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


# --------------------------------------------------------------------------- #
#  Text classifier
# --------------------------------------------------------------------------- #
class TextClassifierHybrid(nn.Module):
    """Transformer‑based text classifier supporting quantum sub‑modules."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_wires: int = 4,
        n_qubits_ffn: int = 4,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum_attention,
                    use_quantum_ffn,
                    n_wires,
                    n_qubits_ffn,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 2
            else nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifierHybrid",
]
