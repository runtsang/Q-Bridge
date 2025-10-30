"""Classical regression model with transformer integration."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a superposition‑like signal."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns state tensors and labels."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalEncoder(nn.Module):
    """Linear encoder followed by a sinusoidal mapping that mimics quantum amplitudes."""
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(num_features, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # Mimic quantum amplitude mapping
        x = torch.sin(x) + 0.1 * torch.cos(2 * x)
        return self.dropout(x)

class TransformerBlockClassical(nn.Module):
    """Standard transformer block using multi‑head attention and a feed‑forward network."""
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

class QuantumRegressionEnhanced(nn.Module):
    """Hybrid regression model that optionally concatenates a transformer stack."""
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.encoder = ClassicalEncoder(num_features, embed_dim)
        if use_transformer:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.transformers(x)
        x = self.head(x)
        return x.squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "ClassicalEncoder",
    "TransformerBlockClassical",
    "QuantumRegressionEnhanced",
]
