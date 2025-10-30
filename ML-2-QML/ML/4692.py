"""Hybrid classical model that fuses CNN features, an auto‑encoder bottleneck, and a quantum‑style drop‑in FCL."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class QuantumNATGen(nn.Module):
    """
    Classical counterpart of the hybrid QuantumNAT model.
    Architecture:
        * 2×Conv + 2×Pool for 28×28 grayscale images
        * Fully‑connected encoder to a latent vector
        * Optional FCL layer (drop‑in for quantum version)
        * Fully‑connected decoder to 4 outputs
        * BatchNorm on the final output
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Convolutional encoder
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Flattened feature dimension: 16 * 7 * 7
        self._flat_dim = 16 * 7 * 7

        # Classical auto‑encoder bottleneck
        encoder_layers = [nn.Linear(self._flat_dim, hidden_dims[0]), nn.ReLU(inplace=True)]
        encoder_layers += [nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
        encoder_layers += [nn.Linear(hidden_dims[0], latent_dim), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*encoder_layers)

        # Optional fully‑connected layer that mimics the quantum FCL
        self.fcl = nn.Linear(latent_dim, 1)

        # Decoder to output 4 features
        decoder_layers = [
            nn.Linear(1, hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], 4),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)           # (bsz, 16, 7, 7)
        flat = feats.view(bsz, -1)         # (bsz, 16*7*7)
        latent = self.encoder(flat)        # (bsz, latent_dim)
        fcl_out = self.fcl(latent)         # (bsz, 1)
        decoded = self.decoder(fcl_out)    # (bsz, 4)
        return self.norm(decoded)

__all__ = ["QuantumNATGen"]
