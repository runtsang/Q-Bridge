"""Hybrid transformer classifier integrating optional quantum attention.

This module extends the original classical transformer with a lightweight
wrapper around the SelfAttention circuit (from the SelfAttention.py seed).
The class can be instantiated with purely classical sub‑modules or with
quantum‑style attention/FFN blocks, enabling controlled ablation studies
without changing the overall API.

The design follows the anchor QTransformerTorch.py but adds the
`use_quantum_attention` and `use_quantum_ffn` flags.  No external
quantum libraries are required for the classical path; the quantum
attention is executed on the CPU via NumPy, keeping the dependency
footprint minimal.
"""

from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical SelfAttention helper that mimics a quantum API
# The helper is a factory that returns a class instance.
from SelfAttention import SelfAttention as ClassicalSelfAttentionFactory


# --------------------------------------------------------------------------- #
# Helper modules
# --------------------------------------------------------------------------- #
class QuantumSelfAttentionWrapper(nn.Module):
    """
    Wraps the classical `SelfAttention` helper to behave like a PyTorch
    attention module.  Parameters are learnable tensors that are reshaped
    into the rotation and entangle matrices expected by the circuit.
    """

    def __init__(self, embed_dim: int):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the token embeddings.  The helper assumes a
            4‑dimensional space; we therefore enforce `embed_dim % 4 == 0`.
        """
        super().__init__()
        if embed_dim % 4!= 0:
            raise ValueError("SelfAttention helper is implemented for embed_dim divisible by 4")
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(
            torch.randn(embed_dim, embed_dim // 4, dtype=torch.float32)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(embed_dim, embed_dim // 4, dtype=torch.float32)
        )
        self.helper = ClassicalSelfAttentionFactory(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum‑style self‑attention circuit.

        The input `x` should be of shape `(batch, seq_len, embed_dim)`.
        The helper operates on a 2‑D array `(batch * seq_len, embed_dim)`,
        so we reshape, call the helper, then reshape back.
        """
        batch, seq, dim = x.size()
        flat = x.reshape(batch * seq, dim).cpu().numpy()
        out_np = self.helper.run(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            flat,
        )
        out = torch.from_numpy(out_np).to(x.device).float()
        return out.reshape(batch, seq, dim)


# --------------------------------------------------------------------------- #
# Core transformer blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Alias for API compatibility – the quantum variant is implemented in the
    QML module.
    """
    pass


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
    """
    Alias for API compatibility – the quantum variant is implemented in
    the QML module.
    """
    pass


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockClassical):
    """
    Alias for API compatibility – the quantum variant is implemented in
    the QML module.
    """
    pass


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
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
# Main model
# --------------------------------------------------------------------------- #
class HybridTransformerClassifier(nn.Module):
    """
    Transformer‑based text classifier that can swap between classical
    and quantum sub‑modules.  The quantum components are lightweight
    wrappers around the SelfAttention circuit or TorchQuantum blocks,
    enabling controlled experiments without incurring the full
    simulation overhead.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
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
    use_quantum_attention : bool, optional
        If True, replace the classical attention with the
        `QuantumSelfAttentionWrapper`.
    use_quantum_ffn : bool, optional
        If True, replace the classical feed‑forward with the quantum
        variant defined in the QML module.
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
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Build transformer blocks
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_attention:
                attn = QuantumSelfAttentionWrapper(embed_dim)
            else:
                attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

            if use_quantum_ffn:
                ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)
            else:
                ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

            block = TransformerBlockBase(embed_dim, num_heads, dropout)
            # Monkey‑patch the attention and feed‑forward parts
            block.attn = attn
            block.ffn = ffn
            blocks.append(block)

        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "HybridTransformerClassifier",
    "QuantumSelfAttentionWrapper",
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
]
