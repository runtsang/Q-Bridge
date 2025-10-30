"""HybridTransformer: classical transformer with optional quantum configuration.

This module implements a pure‑classical transformer that mirrors the API of the
quantum variant.  The constructor accepts the same keyword arguments as the
quantum implementation, but the quantum parameters are ignored.  The class
provides a simple text classifier/regressor and a lightweight dataset
generator inspired by the regression example.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics a quantum superposition.

    Parameters
    ----------
    num_features
        Dimensionality of the input vectors.
    samples
        Number of samples to generate.

    Returns
    -------
    x
        Uniformly distributed features in ``[-1, 1]``.
    y
        Non‑linear target: ``sin(sum(x)) + 0.1*cos(2*sum(x))``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset based on ``generate_superposition_data``."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

# --------------------------------------------------------------------------- #
# Transformer primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared interface for attention layers."""

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
    """Standard multi‑head attention implemented with PyTorch ops."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


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
    """Two‑layer MLP with ReLU."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


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
    """Standard transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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
# Hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Hybrid transformer that exposes the same API for classical and quantum back‑ends.

    Parameters
    ----------
    vocab_size
        Number of tokens in the vocabulary.
    embed_dim
        Dimensionality of token embeddings.
    num_heads
        Number of attention heads.
    num_blocks
        Number of transformer layers.
    ffn_dim
        Dimensionality of the intermediate MLP.
    num_classes
        Number of output classes; ``1`` indicates a regression task.
    dropout
        Drop‑out probability.
    task_type
        ``"classification"`` or ``"regression"``.
    n_qubits_transformer
        If >0 the quantum transformer variant will be used (ignored by the
        classical implementation).
    n_qubits_ffn
        Number of qubits for the quantum feed‑forward sub‑module (ignored
        by the classical implementation).
    n_qubits_head
        Number of qubits for a quantum regression head (ignored by the
        classical implementation).
    q_device
        Optional pre‑instantiated quantum device (ignored by the classical
        implementation).
    n_qlayers
        Number of quantum layers in the attention head (ignored by the
        classical implementation).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int = 1,
        dropout: float = 0.1,
        task_type: str = "classification",
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qubits_head: int = 0,
        q_device: Optional[object] = None,
        n_qlayers: int = 1,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        # Build transformer layers
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        # Head
        if task_type == "regression":
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and classify/regress a batch of token IDs.

        Parameters
        ----------
        x
            ``(batch, seq_len)`` tensor of token indices.
        """
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        x = self.transformers(x)
        # Pooling: mean over sequence dimension
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)

__all__ = [
    "HybridTransformer",
    "generate_superposition_data",
    "RegressionDataset",
]
