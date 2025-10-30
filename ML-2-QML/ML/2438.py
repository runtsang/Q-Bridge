"""HybridTransformer: classical transformer with optional quantum submodules.

This module merges the classical transformer implementation from
QTransformerTorch.py with the quantum variants from the QML seed.
The class exposes a `use_quantum` flag that lazily loads the quantum
implementations when needed.  A lightweight `FCL` helper is also
provided, mirroring the fully‑connected layer example from FCL.py.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Base classes shared by both classical and quantum variants
# ----------------------------------------------------------------------
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

# Quantum variant is a lightweight alias – real implementation lives in the QML module
class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias for API compatibility – actual quantum implementation is loaded lazily."""

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
    """Alias for API compatibility – real quantum implementation is loaded lazily."""

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
    """Alias for API compatibility – real quantum implementation is loaded lazily."""

# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Fully‑connected layer helper (classical)
# ----------------------------------------------------------------------
def FCL():
    """Return a classical fully‑connected layer with a ``run`` method."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()

# ----------------------------------------------------------------------
# Hybrid transformer
# ----------------------------------------------------------------------
class HybridTransformer(nn.Module):
    """A transformer that can switch between classical and quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimension of the token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of output classes (1 for binary classification).
    dropout : float, optional
        Drop‑out probability.
    use_quantum : bool, default False
        If True, the transformer will attempt to load the quantum
        implementation from the QML module.  When quantum sub‑modules
        are requested but the QML module is unavailable, the class falls
        back to the classical implementation.
    q_device : optional
        Quantum device to be used by the QML implementation.
    n_qubits_transformer : int, default 0
        Number of qubits for the quantum transformer attention heads.
    n_qubits_ffn : int, default 0
        Number of qubits for the quantum feed‑forward layer.
    n_qlayers : int, default 1
        Number of quantum layers to stack inside the transformer block.
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
        q_device=None,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        if self.use_quantum and (n_qubits_transformer > 0 or n_qubits_ffn > 0):
            try:
                # Lazy import to avoid heavy dependencies when the
                # quantum implementation is not required.
                from.QTransformerTorch__gen245_qml import HybridTransformer as QuantumHybrid
                quantum = QuantumHybrid(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_blocks,
                    ffn_dim,
                    num_classes,
                    dropout,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    q_device,
                )
                self.transformers = quantum.transformers
                self.classifier = quantum.classifier
                self.dropout = quantum.dropout
            except Exception as exc:
                # Fall back to classical implementation if import fails.
                print(f"Quantum implementation unavailable: {exc}. Falling back to classical.")
                self.transformers = nn.Sequential(
                    *[
                        TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                        for _ in range(num_blocks)
                    ]
                )
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
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
    "FCL",
    "HybridTransformer",
]
