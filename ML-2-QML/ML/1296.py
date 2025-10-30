"""Hybrid transformer implementation for classical training pipelines.

The module keeps the original API surface but expands the
architecture to support a *hybrid* mode where only the
attention heads are quantum‑enabled while the feed‑forward
network stays classical.  The design is intentionally
minimal so that existing training loops can be reused
without modification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum modules; they are defined in the companion QML module.
# The import is delayed to avoid heavy dependencies when the user only
# needs the classical path.
try:
    from.QTransformerTorch__gen248_qml import QuantumAttention, QuantumFeedForward
except Exception:  # pragma: no cover
    # If the QML module cannot be imported (e.g. missing pennylane),
    # we fall back to None and raise an informative error only when
    # quantum functionality is requested.
    QuantumAttention = None  # type: ignore
    QuantumFeedForward = None  # type: ignore


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, E) -> (B, H, T, d_k)."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention with optional mask."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Purely classical multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that maps projections through a variational quantum circuit."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits: int = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if QuantumAttention is None:
            raise ImportError("QuantumAttention module not available; install pennylane.")
        self.n_qubits = n_qubits or self.d_k
        self.q_attention = QuantumAttention(self.n_qubits, self.d_k)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        # Flatten heads for batched quantum evaluation
        def quantum_transform(tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(batch_size * self.num_heads, seq_len, self.d_k)
            out = self.q_attention(flat)
            return out.reshape(batch_size, self.num_heads, seq_len, self.d_k)

        q = quantum_transform(q)
        k = quantum_transform(k)
        v = quantum_transform(v)

        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a variational quantum circuit."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1,
                 n_qubits: int = None) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        if QuantumFeedForward is None:
            raise ImportError("QuantumFeedForward module not available; install pennylane.")
        self.n_qubits = n_qubits or ffn_dim
        self.q_ffn = QuantumFeedForward(self.n_qubits, ffn_dim)
        self.linear1 = nn.Linear(ffn_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.q_ffn(x)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockHybrid(TransformerBlockBase):
    """Transformer block that can mix classical and quantum sub‑modules."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 n_qubits_attention: int = 0,
                 n_qubits_ffn: int = 0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits=n_qubits_attention)
            if use_quantum_attention
            else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, dropout, n_qubits=n_qubits_ffn)
            if use_quantum_ffn
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_emb(pos_ids)


class TextClassifier(nn.Module):
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
        n_qubits_attention: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum_attention,
                    use_quantum_ffn,
                    n_qubits_attention,
                    n_qubits_ffn,
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
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
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
