"""Hybrid classical regression model combining autoencoder, transformer, and kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Data generation (same as anchor)
def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression data from a superposition‑like function."""
    x = torch.rand(samples, num_features) * 2 - 1  # uniform in [-1,1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": self.features[idx], "target": self.labels[idx]}

# Autoencoder
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout)])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout)])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# Transformer
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Kernel (classical RBF)
class Kernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# Main regression model
class QuantumRegressionModel(nn.Module):
    """Hybrid regression model combining autoencoder, transformer, and kernel."""
    def __init__(self, num_features: int, latent_dim: int = 32,
                 transformer_blocks: int = 2, num_heads: int = 4,
                 ffn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.autoencoder = AutoencoderNet(AutoencoderConfig(num_features, latent_dim))
        self.pos_encoder = PositionalEncoder(latent_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(latent_dim, num_heads, ffn_dim, dropout)
            for _ in range(transformer_blocks)
        ])
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)          # latent representation
        z = z.unsqueeze(1)                     # sequence length 1
        z = self.pos_encoder(z)
        z = self.transformer(z)
        z = z.mean(dim=1)                       # global pooling
        return self.regressor(z).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset",
           "generate_superposition_data", "Kernel"]
