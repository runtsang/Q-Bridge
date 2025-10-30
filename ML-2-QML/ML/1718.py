"""
QTransformerExtended: classical transformer enriched with advanced control flow.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """
    Base class for multi‑head attention with optional trainable mask.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_mask: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        if use_mask:
            # Trainable mask applied to attention logits
            self.mask = nn.Parameter(torch.ones(1, 1, embed_dim, embed_dim))

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dimension into multiple heads.
        """
        batch, seq, _ = x.shape
        return (
            x.view(batch, seq, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention with optional trainable mask.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_mask: bool = False):
        super().__init__(embed_dim, num_heads, dropout, use_mask)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.use_mask:
            scores = scores * self.mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForwardClassical(nn.Module):
    """
    Two‑layer perceptron feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardGated(nn.Module):
    """
    Feed‑forward with a sigmoid gate that modulates the output.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = F.relu(h)
        h = self.linear2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(x))
        return h * g


class TransformerBlockExtended(nn.Module):
    """
    Transformer block that can optionally use a gated feed‑forward
    and a trainable attention mask.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_gate: bool = False,
        use_mask: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout, use_mask)
        self.ffn = FeedForwardGated(embed_dim, ffn_dim, dropout) if use_gate else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
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


class QTransformerExtended(nn.Module):
    """
    Transformer‑based text classifier that supports both a classical
    and a quantum‑augmented architecture.
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
        use_gate: bool = False,
        use_mask: bool = False,
        weight_sharing: bool = False,
        n_qubits: Optional[int] = None,
        n_qlayers: int = 1,
        q_device: Optional[object] = None,  # type: ignore
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.use_gate = use_gate
        self.use_mask = use_mask
        self.weight_sharing = weight_sharing

        # Decide on classical or quantum transformer
        if n_qubits is None or n_qubits <= 0:
            # Classical transformer
            block_cls = TransformerBlockExtended
            blocks = [
                block_cls(embed_dim, num_heads, ffn_dim, dropout, use_gate, use_mask)
                for _ in range(num_blocks)
            ]
            self.transformer = nn.Sequential(*blocks)
        else:
            # Placeholder: quantum blocks will be defined in the qml_code module
            # The class will be replaced by the quantum implementation
            # at runtime if available.
            self.transformer = None  # type: ignore

        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If the transformer is None, it means the quantum
        implementation will be used. The quantum implementation should
        provide a compatible forward signature.
        """
        if self.transformer is None:
            raise RuntimeError("Quantum transformer not initialized. "
                               "Load the quantum implementation before calling forward.")
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def set_params(
        self,
        *,
        dropout: Optional[float] = None,
        use_gate: Optional[bool] = None,
        use_mask: Optional[bool] = None,
        weight_sharing: Optional[bool] = None,
    ) -> None:
        """
        Dynamically update hyperparameters that affect the transformer blocks.
        """
        if dropout is not None:
            self.dropout.p = dropout
            for block in getattr(self, "transformer", []):
                if hasattr(block, "attn"):
                    block.attn.dropout.p = dropout
                if hasattr(block, "ffn"):
                    block.ffn.dropout.p = dropout
        if use_gate is not None:
            self.use_gate = use_gate
        if use_mask is not None:
            self.use_mask = use_mask
        if weight_sharing is not None:
            self.weight_sharing = weight_sharing

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "FeedForwardGated",
    "TransformerBlockExtended",
    "PositionalEncoder",
    "QTransformerExtended",
]
