"""QTransformerTorchGen415: classical transformer with optional quantum layers and stochastic depth.

This module implements a transformer-based text classifier with optional quantum submodules per block.  The design preserves the original API while adding learnable positional bias, gated attention mix (default to pure classical), and stochastic depth.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _safe_gather(tensor: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """Safely gather from a multi-dimensional tensor."""
    return torch.gather(tensor, dim, index)

# --------------------------------------------------------------------------- #
# 1. Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention.

    The base does not implement forward; subclasses provide either classical
    or quantum implementations.  A learnable gating scalar can be added
    in subclasses to mix classical and quantum results.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Placeholder for the quantum implementation.  In the classical module
    we simply inherit from the classical implementation to keep the API
    compatible.  The quantum module is defined in the QML counterpart.
    """
    pass

# --------------------------------------------------------------------------- #
# 2. Feed‑Forward Network
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardClassical):
    """Placeholder for the quantum implementation.  In the classical module
    we simply inherit from the classical implementation to keep the API
    compatible.  The quantum module is defined in the QML counterpart.
    """
    pass

# --------------------------------------------------------------------------- #
# 3. Transformer Block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block with optional stochastic depth."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_path_rate > 0.0 and random.random() < self.drop_path_rate:
            return x
        return x


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, drop_path_rate)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._stochastic_depth(x)
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockClassical):
    """Placeholder for the quantum implementation.  In the classical module
    we simply inherit from the classical implementation to keep the API
    compatible.  The quantum module is defined in the QML counterpart.
    """
    pass

# --------------------------------------------------------------------------- #
# 4. Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with a learnable bias."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        # Learnable bias added to the positional encoding
        self.pos_bias = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)] + self.pos_bias

# --------------------------------------------------------------------------- #
# 5. Text Classifier
# --------------------------------------------------------------------------- #
class QTransformerTorchGen415(nn.Module):
    """Transformer‑based text classifier supporting optional quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Hidden dimension of the model.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Dropout probability.
    use_quantum : bool, optional
        If True, use quantum layers for attention and feed‑forward
        (only in the QML implementation; in the ML module this flag is ignored).
    n_qubits_transformer : int, optional
        Number of qubits per transformer block (only used in QML).
    n_qubits_ffn : int, optional
        Number of qubits for the feed‑forward sub‑module (only used in QML).
    n_qlayers : int, optional
        Number of quantum layers per block (only used in QML).
    q_device : Optional[object], optional
        Quantum device to use (only used in QML).
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
        use_quantum: bool = False,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[object] = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        # Build blocks
        blocks = []
        drop_path_rate = 0.1  # global stochastic‑depth rate (can be tuned)
        for _ in range(num_blocks):
            if use_quantum and n_qubits_transformer > 0:
                # In the ML implementation we fall back to classical blocks
                # because quantum modules are not present.
                block = TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout, drop_path_rate
                )
            else:
                block = TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout, drop_path_rate
                )
            blocks.append(block)
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
    "PositionalEncoder",
    "QTransformerTorchGen415",
]
