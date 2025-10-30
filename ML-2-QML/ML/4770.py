"""Hybrid regression model with classical back‑bone.

The public class ``HybridRegression`` can be instantiated with
``use_quantum=False`` (default) and will use a classic auto‑encoder
followed by a transformer encoder.  The module also exposes the
dataset and data‑generation logic for consistency with the quantum
counterpart.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(samples, num_features).float() * 2 - 1  # uniform on [-1, 1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": self.features[idx], "target": self.labels[idx]}

# ----- classical sub‑modules -----
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(*[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class HybridRegression(nn.Module):
    """Classical hybrid regression model."""
    def __init__(
        self,
        num_features: int,
        *,
        num_heads: int = 4,
        ffn_dim: int = 64,
        transformer_layers: int = 2,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.use_quantum = False
        self.autoencoder = AutoencoderNet(num_features, latent_dim)
        self.transformer = TransformerEncoder(num_features, num_heads, ffn_dim, transformer_layers)
        self.head = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        t = self.transformer(z)
        return self.head(t).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
