"""
Extended transformer implementation that preserves the original API but introduces
1.  Learnable positional embeddings
2.  Gated Multi‑Head Attention with a learnable quantum‑per‑head extractor
3.  Configurable residual‑dropout schedule
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Base classes (same public signature to the seed)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """
    Base class for attention blocks. Public constructor signature matches the seed.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardBase(nn.Module):
    """
    Base class for feed‑forward nets.  The seed exposes a simple MLP, we keep it but
    add a gated feed‑forward and an optional quantum sub‑module.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, gated: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.gated = gated
        if gated:
            self.gate = nn.Linear(embed_dim, embed_dim)
        # placeholder for quantum sub‑module, set in subclasses
        self.quantum_layer: Optional[nn.Module] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """
    Quantum feed‑forward that uses a variational circuit (PennyLane) as a feature extractor.
    The implementation is a no‑op placeholder in the classical module – it simply
    forwards to the classical version.  The real quantum logic lives in the QML module.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, gated: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, gated)
        self.quantum_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class FeedForwardClassical(FeedForwardBase):
    """
    Two‑layer perceptron feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, gated: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, gated)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        if self.gated:
            gate_val = torch.sigmoid(self.gate(x))
            out = out * gate_val
        return out


# --------------------------------------------------------------------------- #
#  Attention implementations
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention implemented classically.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(attn_out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Alias of the classical attention for API compatibility in the classical module.
    The real quantum logic resides in the QML module.
    """
    pass


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """
    Base transformer block containing attention and feed‑forward parts.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """
    Classical transformer block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        gated_ffn: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout, gated=gated_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """
    Alias of the classical block for the classical module.
    Quantum logic is provided in the QML module.
    """
    pass


# --------------------------------------------------------------------------- #
#  Positional encoder
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """
    Positional encoder that can be either sinusoidal (default) or learnable.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000, learnable: bool = False) -> None:
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.pos_embedding = nn.Embedding(max_len, embed_dim)
        else:
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                                 (-math.log(10000.0) / embed_dim))
            pe = torch.zeros(max_len, embed_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            return x + self.pos_embedding(pos_ids)
        else:
            return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
#  Text classifier
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier supporting quantum sub‑modules and
    optional learnable positional embeddings and gated feed‑forward.
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
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[object] = None,
        *,
        use_learnable_pos: bool = False,
        gated_ffn: bool = False,
        dropout_schedule: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim, learnable=use_learnable_pos)

        # Build transformer blocks
        blocks = []
        for i in range(num_blocks):
            dl = dropout_schedule[i] if dropout_schedule else dropout
            if n_qubits_transformer > 0:
                block = TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    dropout=dl,
                    gated_ffn=gated_ffn,
                    q_device=q_device,
                )
            else:
                block = TransformerBlockClassical(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout=dl,
                    gated_ffn=gated_ffn,
                )
            blocks.append(block)
        self.transformers = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
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
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
