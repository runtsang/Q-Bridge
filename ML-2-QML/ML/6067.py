"""Hybrid transformer implementation combining classical multi‑head attention and a lightweight
Self‑Attention wrapper that mirrors the quantum API.

The module preserves the public API of the original ``QTransformerTorch`` while adding a
``SelfAttention`` block that can be swapped in place of the standard attention mechanism.
The implementation is fully classical (PyTorch) and is ready to be used as a drop‑in
replacement in existing pipelines.

Key extensions
---------------
* ``SelfAttention`` – a pure‑Python self‑attention helper that accepts learnable rotation
  and entanglement parameters.
* ``HybridTransformer`` – accepts an ``attention_type`` argument to choose between
  standard multi‑head attention and the self‑attention block.
* ``__all__`` lists all public symbols for convenient imports.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Self‑Attention helper (classical, API‑compatible with the quantum version)
# --------------------------------------------------------------------------- #
class SelfAttention:
    """Simple self‑attention utility that mimics the quantum API.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.

    The class exposes ``run`` which expects rotation and entanglement parameters
    together with an input tensor of shape ``(batch, seq_len, embed_dim)``.
    """

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Convert to tensors for efficient computation
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
#  Attention abstractions
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention implementations."""

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
    """Standard multi‑head attention implemented with PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias of the classical attention for API symmetry."""
    pass


# --------------------------------------------------------------------------- #
#  Feed‑forward abstractions
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
    """Two‑layer MLP."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardClassical):
    """Alias of the classical feed‑forward block."""
    pass


# --------------------------------------------------------------------------- #
#  Transformer block abstractions
# --------------------------------------------------------------------------- #
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
    """Standard transformer block using classical attention."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockClassical):
    """Alias of the classical block for API symmetry."""
    pass


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Transformer that can swap between classical multi‑head attention and a
    lightweight SelfAttention block.  The design mirrors the API of the original
    ``QTransformerTorch`` so existing code can be reused with minimal changes.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the token embedding.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Hidden dimension inside the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float
        Drop‑out probability.
    attention_type : str, default "classical"
        Either ``"classical"`` to use the standard multi‑head attention or
        ``"self_attention"`` to use the SelfAttention helper.
    rotation_dim : int, optional
        Dimensionality of the rotation matrix for the SelfAttention block
        (only used when ``attention_type=="self_attention"``).
    entangle_dim : int, optional
        Dimensionality of the entanglement matrix for the SelfAttention block
        (only used when ``attention_type=="self_attention"``).
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
        attention_type: str = "classical",
        rotation_dim: Optional[int] = None,
        entangle_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Decide on the attention implementation
        if attention_type == "self_attention":
            if rotation_dim is None or entangle_dim is None:
                raise ValueError("rotation_dim and entangle_dim must be provided for self_attention")
            self.attn_block = SelfAttentionBlock(
                embed_dim=embed_dim,
                rotation_dim=rotation_dim,
                entangle_dim=entangle_dim,
            )
        else:
            self.attn_block = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

        # Build transformer layers
        self.transformers = nn.ModuleList(
            [
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)

        # Apply attention block first (optional)
        if isinstance(self.attn_block, SelfAttentionBlock):
            # SelfAttention expects numpy inputs
            rotation_params = self.attn_block.rotation_params
            entangle_params = self.attn_block.entangle_params
            x_np = x.detach().cpu().numpy()
            x_np = self.attn_block.run(rotation_params, entangle_params, x_np)
            x = torch.from_numpy(x_np).to(x.device).float()
        else:
            x = self.attn_block(x)

        # Follow with transformer blocks
        for block in self.transformers:
            x = block(x)

        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


class SelfAttentionBlock(nn.Module):
    """Wrapper around the pure‑Python ``SelfAttention`` helper.

    The class exposes learnable rotation and entanglement parameters that
    are passed to the underlying helper during the forward pass.
    """

    def __init__(self, embed_dim: int, rotation_dim: int, entangle_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(
            torch.randn(rotation_dim * embed_dim)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(entangle_dim * embed_dim)
        )
        self.helper = SelfAttention(embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        return self.helper.run(rotation_params, entangle_params, inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward is handled in the HybridTransformer; this method is not used
        return x


__all__ = [
    "SelfAttention",
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
    "HybridTransformer",
    "SelfAttentionBlock",
]
