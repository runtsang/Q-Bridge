from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the lightweight auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with configurable depth."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*self._build_layers(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout))
        self.decoder = nn.Sequential(*self._build_layers(cfg.latent_dim, tuple(reversed(cfg.hidden_dims)), cfg.input_dim, cfg.dropout))

    @staticmethod
    def _build_layers(in_dim: int, hidden: Tuple[int,...], out_dim: int, dropout: float) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

class EstimatorQNN__gen044(nn.Module):
    """
    Classical hybrid model: a variational auto‑encoder followed by a linear regressor.
    The auto‑encoder reduces dimensionality; the regressor predicts a scalar output.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        self.autoencoder = AutoencoderNet(cfg)
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.regressor(z)

__all__ = ["EstimatorQNN__gen044"]
