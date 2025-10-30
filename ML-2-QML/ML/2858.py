"""Unified classical regression transformer.

The model processes a 1‑D feature vector as a sequence of tokens,
passes them through a standard transformer, and outputs a scalar
regression value.  It mirrors the quantum seed by providing a
matching dataset generator and a regression head, but the network
is fully classical and compatible with PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sinusoidal targets from a linear combination of features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding a feature vector and a scalar target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding used by the transformer."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerBlockClassical(nn.Module):
    """Standard transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
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

class UnifiedRegressionTransformer(nn.Module):
    """Transformer‑based regression model that can be configured to use
    either a purely classical transformer or a quantum‑enhanced one.
    The quantum variant is implemented in the separate QML module.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(num_features, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Shape (batch, num_features)

        Returns
        -------
        torch.Tensor
            Shape (batch,)
        """
        x = self.embed(states)          # (batch, num_features, embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)               # global average pooling
        return self.head(x).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "UnifiedRegressionTransformer",
    "PositionalEncoder",
    "TransformerBlockClassical",
]
