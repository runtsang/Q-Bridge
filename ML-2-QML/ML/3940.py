"""Hybrid classical autoencoder with a quantum‑inspired layer placeholder."""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    """Classic autoencoder that can be extended with a quantum layer."""
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            dims=list(config.hidden_dims) + [config.latent_dim],
            dropout=config.dropout,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            dims=list(reversed(config.hidden_dims)) + [config.input_dim],
            dropout=config.dropout,
        )

    def _build_mlp(self, in_dim: int, dims: list[int], dropout: float):
        layers = []
        cur = in_dim
        for d in dims:
            layers.append(nn.Linear(cur, d))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            cur = d
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    config = HybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(config)

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "HybridConfig"]
