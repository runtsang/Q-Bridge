import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data from a superposition-inspired distribution.
    Returns a feature tensor (samples, num_features) and a target tensor (samples,).
    """
    x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping the synthetic data generator.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # pragma: no cover
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {
            "states": self.features[idx],
            "target": self.labels[idx],
        }


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding compatible with transformer inputs.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttentionClassical(nn.Module):
    """
    Standard multi‑head attention used in the classical transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        attn_out, _ = self.attn(x, x, x)
        return self.dropout(attn_out)


class FeedForwardClassical(nn.Module):
    """
    Two‑layer perceptron feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.net(x)


class TransformerBlockClassical(nn.Module):
    """
    Classical transformer encoder block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class ScaleShift(nn.Module):
    """
    Learnable scale and shift applied after transformer sequence.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class HybridQuantumRegression(nn.Module):
    """
    Classical regression model inspired by transformer and photonic scaling.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(num_features, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.ModuleList(
            [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.scale_shift = ScaleShift()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for block in self.transformer:
            x = block(x)
        x = self.scale_shift(x)
        x = self.head(x)
        return x.squeeze(-1)


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "HybridQuantumRegression",
]
