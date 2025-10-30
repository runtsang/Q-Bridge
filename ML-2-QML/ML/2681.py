"""Hybrid estimator autoencoder combining classical MLP encoder‑decoder with a regression head.

This module provides a lightweight PyTorch implementation that mirrors the
classical EstimatorQNN and Autoencoder seeds.  The network first
encodes the input into a latent space, decodes it back for reconstruction,
and then passes the latent representation through a small regressor to
produce a scalar output.  The architecture is fully differentiable and
can be trained end‑to‑end with standard optimizers.

The class name `HybridEstimatorAutoencoder` is intentionally shared with
the quantum implementation to allow a seamless switch between the two
backends in downstream code.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder.append(nn.Linear(in_dim, hidden))
            encoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder.append(nn.Linear(in_dim, hidden))
            decoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

class HybridEstimatorAutoencoder(nn.Module):
    """Hybrid classical estimator that uses an autoencoder latent space
    followed by a small regression head."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(cfg)
        self.regressor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.autoencoder.encode(x)
        return self.regressor(latent)

def HybridEstimatorAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridEstimatorAutoencoder:
    """Convenience factory mirroring the original EstimatorQNN signature."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridEstimatorAutoencoder(cfg)

__all__ = ["HybridEstimatorAutoencoder", "HybridEstimatorAutoencoderFactory", "AutoencoderConfig", "AutoencoderNet"]
