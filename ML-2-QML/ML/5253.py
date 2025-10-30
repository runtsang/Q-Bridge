"""Hybrid regression model – classical implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------- #
# Data generation (mirrors the original seed)
# --------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression data."""
    x = torch.rand(samples, num_features) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    """Simple dataset wrapping the synthetic data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": self.features[index],
            "target": self.labels[index],
        }

# --------------------------------------------------------------------- #
# Classical quanvolution filter
# --------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """2×2 patch-based convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # (batch, 4*14*14)

# --------------------------------------------------------------------- #
# Autoencoder (fully‑connected)
# --------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Encoder–decoder architecture."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Convenience factory."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# --------------------------------------------------------------------- #
# Hybrid regression model (classical)
# --------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Full pipeline: quanvolution → autoencoder → MLP."""
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        # 28×28 MNIST‑style input → 4×14×14 features
        self.qfilter = QuanvolutionFilter()
        self.autoencoder = Autoencoder(
            input_dim=4 * 14 * 14,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=0.1,
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass for a batch of images."""
        # Feature extraction
        features = self.qfilter(x)  # (batch, 4*14*14)
        # Compression
        latents = self.autoencoder.encode(features)
        # Regression head
        out = self.regressor(latents)
        return out.squeeze(-1)

__all__ = [
    "HybridRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "QuanvolutionFilter",
    "AutoencoderNet",
    "Autoencoder",
    "AutoencoderConfig",
]
