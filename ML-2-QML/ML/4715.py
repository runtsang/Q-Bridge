"""Hybrid classical model combining autoencoder, sampler, and classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Sampler block
# --------------------------------------------------------------------------- #
class SamplerModule(nn.Module):
    """Softmax sampler operating on the latent space."""
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(z), dim=-1)


# --------------------------------------------------------------------------- #
# Hybrid classifier head
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """End‑to‑end hybrid model: autoencoder → sampler → linear head."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(
            input_dim, latent_dim, hidden_dims, dropout)
        self.sampler = SamplerModule(latent_dim)
        self.head = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.autoencoder.encode(x)
        prob = self.sampler(z)
        return self.head(prob)


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def build_hybrid_model(input_dim: int,
                       latent_dim: int = 32,
                       hidden_dims: Tuple[int,...] = (128, 64),
                       dropout: float = 0.1,
                       num_classes: int = 2) -> HybridClassifier:
    """Return a fully‑initialized hybrid classifier."""
    return HybridClassifier(input_dim, latent_dim, hidden_dims, dropout, num_classes)


__all__ = [
    "AutoencoderNet",
    "SamplerModule",
    "HybridClassifier",
    "build_hybrid_model",
]
