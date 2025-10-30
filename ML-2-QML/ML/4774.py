"""Classical regression module with transformer backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

# --------------------------------------------------------------------------- #
# 1. Dataset & data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic superposition data for regression.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (samples, num_features) with values in [-1, 1].
    y : np.ndarray
        Target vector of shape (samples,) computed as
        sin(Σx_i) + 0.1 * cos(2 Σx_i).
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the deterministic superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# 2. Classical backbone
# --------------------------------------------------------------------------- #

class ClassicalMLP(nn.Module):
    """
    Shallow multi‑layer perceptron used as a baseline predictor.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int] = (32, 16)):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing feature representations of shape (B, H),
        where H = hidden_dims[-1].
        """
        return self.net(x)

# --------------------------------------------------------------------------- #
# 3. Transformer components (classical implementation)
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """
    Base class for multi‑head attention layers.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention implemented with PyTorch linear projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding dimension does not match layer size")
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        d_k = embed_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.combine_heads(out)

class FeedForwardBase(nn.Module):
    """
    Base class for feed‑forward networks.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """
    Two‑layer feed‑forward network with ReLU activation.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    """
    Base class for a transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """
    Classical transformer block consisting of multi‑head attention and feed‑forward.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
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

class TransformerBackbone(nn.Module):
    """
    Sequence of transformer blocks.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# --------------------------------------------------------------------------- #
# 4. Hybrid regression model
# --------------------------------------------------------------------------- #

class QuantumRegression(nn.Module):
    """
    Classical regression model that can employ either an MLP or a transformer backbone.
    The class name mirrors the quantum counterpart in the QML module.
    """
    def __init__(
        self,
        num_features: int,
        architecture: str = "mlp",
        *,
        hidden_dims: List[int] = (32, 16),
        embed_dim: int | None = None,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        if architecture == "mlp":
            self.backbone = ClassicalMLP(num_features, hidden_dims)
            self.out_dim = hidden_dims[-1]
        elif architecture == "transformer":
            embed_dim = embed_dim or num_features
            self.backbone = TransformerBackbone(embed_dim, num_heads, ffn_dim, num_blocks, dropout)
            self.out_dim = embed_dim
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.head = nn.Linear(self.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "ClassicalMLP",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TransformerBackbone",
    "QuantumRegression",
]
