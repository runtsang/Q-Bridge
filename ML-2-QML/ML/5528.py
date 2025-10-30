"""QTransformerTorch__gen428: Classical transformer with quantum‑compatible API."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Classical building blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention layers."""

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
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.linear_q(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.linear_k(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.linear_v(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block (attention + FFN)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Main transformer classifier
# --------------------------------------------------------------------------- #
class QTransformerTorch__gen428(nn.Module):
    """Hybrid transformer classifier that can be instantiated with classical blocks
    or quantum‑enhanced sub‑modules.

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
        Number of output classes.
    dropout : float, optional
        Dropout probability (default 0.1).
    use_quantum : bool, optional
        If ``True`` the class will raise a ``NotImplementedError`` because the
        quantum implementation lives in the QML module.  The flag exists only
        for API symmetry and to document the intended usage.
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
    ) -> None:
        super().__init__()
        if use_quantum:
            raise NotImplementedError(
                "Quantum configuration is available only in the QML module."
            )
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = num_classes if num_classes > 2 else 1
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global pooling
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Classical wrappers for quantum examples
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    """Return a simple softmax sampler implemented with PyTorch."""

    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


def EstimatorQNN() -> nn.Module:
    """Return a simple regression network implemented with PyTorch."""

    class EstimatorModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 1)
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorModule()


def FCL() -> nn.Module:
    """Return a simple fully‑connected layer that mimics the quantum example."""

    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


__all__ = [
    "QTransformerTorch__gen428",
    "SamplerQNN",
    "EstimatorQNN",
    "FCL",
]
