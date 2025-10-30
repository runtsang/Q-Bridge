"""Hybrid transformer with classical‑quantum interleaving and per‑token gating.

This module extends the original QTransformerTorch API by adding a
TransformerBlockHybrid that blends both classical and quantum
feed‑forward sub‑networks through a learnable per‑token gate.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum feed‑forward module defined in the QML file.
# The QML file must reside in the same directory as this module.
from QTransformerTorch_QML import QuantumFeedForward

class MultiHeadAttentionBase(nn.Module):
    """Base class used by both classical and quantum attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split embedding into multiple heads."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention with optional mask."""
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, S)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.bmm(scores, v), scores

    def downstream(self, q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor, batch_size: int,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project heads back to the original dimension."""
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with torch.nn."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out = nn.Linear(embed_dim, self.embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, _ = self.attention(q, k, v, mask)
        return self.out(out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim))

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias of the classical attention for API compatibility."""
    pass

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Alias of the classical feed‑forward for API compatibility."""
    pass

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

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int,
                 q_device: Optional[object] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockHybrid(nn.Module):
    """Hybrid transformer block that blends classical and quantum
    feed‑forward sub‑networks through a learnable per‑token gate.

    The block uses the standard classical multi‑head attention followed
    by a gating mechanism that chooses between a classical feed‑forward
    network and a quantum feed‑forward network.  The quantum part is
    implemented in :class:`QTransformerTorch_QML.QuantumFeedForward`.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[object] = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Classical sub‑modules
        self.attn_classical = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn_classical = FeedForwardClassical(embed_dim, ffn_dim, dropout)

        # Quantum sub‑modules (only if requested)
        if n_qubits_ffn > 0:
            self.ffn_quantum = QuantumFeedForward(n_qubits_ffn, ffn_dim, dropout, q_device)
        else:
            self.ffn_quantum = None

        # Gate that decides per‑token whether to use the quantum path
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Classical attention
        attn_c = self.attn_classical(x, mask)
        # Residual + norm
        x = self.norm1(x + self.dropout(attn_c))

        # Gate for feed‑forward
        gate = torch.sigmoid(self.gate(x))  # (B, S, 1)

        # Classical feed‑forward
        ffn_c = self.ffn_classical(x)

        # Quantum feed‑forward (if available)
        if self.ffn_quantum is not None:
            ffn_q = self.ffn_quantum(x)
            ffn_out = gate * ffn_q + (1.0 - gate) * ffn_c
        else:
            ffn_out = ffn_c

        # Residual + norm
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting classical‑only,
    quantum‑only, or hybrid configurations.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of target classes.
    dropout : float, default 0.1
        Dropout probability.
    n_qubits_ffn : int, default 0
        Number of qubits for the quantum feed‑forward network.
        If ``0`` the model is purely classical.
    q_device : Optional[object], default None
        Quantum device used by the quantum network.  If ``None``
        a new device is created automatically.
    use_hybrid : bool, default True
        If ``True`` the model uses the hybrid block that blends
        classical and quantum feed‑forward sub‑networks.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[object] = None,
                 use_hybrid: bool = True) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        if use_hybrid:
            blocks = [
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    n_qubits_ffn=n_qubits_ffn,
                    q_device=q_device
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout
                )
                for _ in range(num_blocks)
            ]

        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim,
            num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
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
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
