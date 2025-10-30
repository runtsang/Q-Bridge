"""
Classical implementation of a hybrid regression transformer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data with a sinusoidal target."""
    rng = torch.Generator()
    rng.manual_seed(42)
    x = torch.rand(samples, num_features, generator=rng, dtype=torch.float32) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.features.size(0)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": self.features[idx], "target": self.labels[idx]}

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return self.dropout(scores) @ value, scores

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention with linear projections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.attention(self.separate_heads(q),
                                        self.separate_heads(k),
                                        self.separate_heads(v),
                                        mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(attn_output)

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
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
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class UnifiedRegressionTransformer(nn.Module):
    """
    Hybrid regression transformer that can swap between classical and quantum sub‑modules.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        hidden_size: int | None = None,
        use_quantum: bool = False,
        quantum_config: dict | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size or embed_dim

        # Feature encoder – maps raw features into embed_dim
        self.feature_proj = nn.Linear(num_features, embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Build transformer blocks
        blocks = []
        for _ in range(num_blocks):
            if use_quantum and quantum_config is not None:
                # Import quantum block lazily to avoid circular imports
                from.quantum_unified import QuantumTransformerBlock  # type: ignore
                block = QuantumTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    **quantum_config
                )
            else:
                block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
            blocks.append(block)
        self.transformer = nn.Sequential(*blocks)

        # Output head
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression.
        """
        # Encode features
        x = self.feature_proj(x)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer
        x = self.transformer(x)
        # Pooling – mean over sequence (features act as a sequence)
        x = x.mean(dim=1)
        # Head
        return self.head(x).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "UnifiedRegressionTransformer",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
]
