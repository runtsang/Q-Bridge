"""
Hybrid classical regression model inspired by the original QuantumRegression seed.
It contains:
* a data generator that mimics superposition, but with a richer target signal,
* a small CNN‑style filter that can be used as a drop‑in replacement for a
  quanvolution layer,
* a transformer block with optional quantum attention (currently only the
  classical attention is implemented),
* a final linear head that outputs a scalar regression value.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target is a mixture of sinusoids.
    The function keeps the same interface as the original seed but adds
    an extra cosine term to increase regression difficulty.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.3 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple torch Dataset that returns a dict of state tensors and targets."""
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
# Classical 2‑D convolutional filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# --------------------------------------------------------------------------- #
# Transformer components
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """A single transformer block using the classic MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# The hybrid regression model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(nn.Module):
    """
    Classical regression model that can be swapped with a quantum version
    by importing the module from the QML branch.
    """
    def __init__(self,
                 num_features: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 ffn_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Linear(num_features, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.encoder(batch)
        x = self.transformer(x)
        x = self.head(x).squeeze(-1)
        return x

__all__ = [
    "RegressionDataset",
    "ConvFilter",
    "PositionalEncoding",
    "TransformerBlock",
    "QuantumRegressionModel",
    "generate_superposition_data",
]
