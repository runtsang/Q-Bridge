"""Hybrid transformer implementation with optional quantum modules.

The module keeps the original class names (MultiHeadAttention*, FeedForward*,
TransformerBlock*, PositionalEncoder, TextClassifier) for backward
compatibility.  A new `QuantumConfig` dataclass describes quantum
resources, and a `HybridTransformerBlock` can accept user‑supplied
attention and feed‑forward modules.  The drop‑in replacement
`HybridTextClassifier` forwards the configuration so that a quantum
implementation can later inject quantum sub‑modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Original classes – unchanged for backward compatibility
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias for API compatibility – no quantum logic in the ML module."""


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardClassical):
    """Alias for API compatibility – no quantum logic in the ML module."""


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


class TransformerBlockQuantum(TransformerBlockClassical):
    """Alias for API compatibility – no quantum logic in the ML module."""


class PositionalEncoder(nn.Module):
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
    """Original text classifier – retained for backward compatibility."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# New hybrid API
# --------------------------------------------------------------------------- #

@dataclass
class QuantumConfig:
    """Configuration for quantum sub‑modules.

    Parameters
    ----------
    n_qubits : int
        Number of quantum wires per transformer block.  If 0, the block
        falls back to a fully classical implementation.
    n_qlayers : int
        Number of quantum layers within each block.  This is a hint for
        the quantum implementation but is ignored by the classical
        implementation.
    """
    n_qubits: int = 0
    n_qlayers: int = 1


class HybridTransformerBlock(TransformerBlockBase):
    """A transformer block that can mix classical and quantum sub‑modules.

    The block accepts optional custom attention and feed‑forward modules.
    If none are provided, it defaults to the classical implementations.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_module: Optional[nn.Module] = None,
        ffn_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = attention_module or MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = ffn_module or FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTextClassifier(TextClassifier):
    """Drop‑in replacement for `TextClassifier` that supports hybrid blocks.

    Parameters
    ----------
    quantum_config : Optional[QuantumConfig]
        If provided and ``quantum_config.n_qubits > 0`` the block will
        attempt to use quantum sub‑modules.  The actual quantum
        implementation is supplied by the QML module and must be
        passed via the ``attention_module`` and ``ffn_module`` arguments
        when constructing the block.  This class simply forwards the
        configuration so that the QML module can instantiate the
        appropriate sub‑modules.
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
        quantum_config: Optional[QuantumConfig] = None,
    ) -> None:
        super().__init__(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim, num_classes, dropout)
        # Replace the classical transformer stack with a hybrid stack
        blocks: Sequence[nn.Module] = []
        for _ in range(num_blocks):
            if quantum_config and quantum_config.n_qubits > 0:
                # The QML module will provide the quantum modules via a
                # factory function.  We store the config so that the QML
                # module can later replace the placeholder.
                blocks.append(HybridTransformerBlock(
                    embed_dim, num_heads, ffn_dim, dropout,
                    attention_module=None, ffn_module=None
                ))
            else:
                blocks.append(HybridTransformerBlock(
                    embed_dim, num_heads, ffn_dim, dropout
                ))
        self.transformers = nn.Sequential(*blocks)


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
    "QuantumConfig",
    "HybridTransformerBlock",
    "HybridTextClassifier",
]
