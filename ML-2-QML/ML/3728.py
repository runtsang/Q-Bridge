"""
Hybrid classical autoencoder that combines convolutional feature extraction with a
drop‑out enabled MLP encoder/decoder.  The network is inspired by the
`QuantumNAT.py` CNN+FC model and the fully‑connected autoencoder in
`Autoencoder.py`.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class HybridConfig:
    """Configuration for :class:`HybridNATAutoencoder`."""
    input_channels: int = 1
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    img_size: Tuple[int, int] = (28, 28)  # default for MNIST‑style data


class HybridNATAutoencoder(nn.Module):
    """Convolutional autoencoder with a classical latent space.

    The encoder consists of two convolutional blocks followed by a fully‑connected
    MLP that projects to ``latent_dim`` units.  The decoder mirrors the
    encoder and finally reconstructs the image via transposed convolutions.
    """

    def __init__(self, cfg: HybridConfig = HybridConfig()) -> None:
        super().__init__()

        # Feature extractor – two conv‑pool blocks
        self.features = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        dummy = torch.zeros(1, cfg.input_channels, *cfg.img_size)
        feat_dim = self.features(dummy).view(1, -1).shape[1]

        # Encoder MLP
        enc_layers = []
        in_dim = feat_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder MLP
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, feat_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Reconstruction head – de‑convolutional layers
        self.recon_head = nn.Sequential(
            nn.Unflatten(1, (16, cfg.img_size[0] // 4, cfg.img_size[1] // 4)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, cfg.input_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.norm = nn.BatchNorm1d(cfg.latent_dim)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of ``x``."""
        bsz = x.shape[0]
        feats = self.features(x).view(bsz, -1)
        latent = self.encoder(feats)
        return self.norm(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from a latent vector."""
        feats = self.decoder(latent)
        return self.recon_head(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoder‑decoder pass."""
        latent = self.encode(x)
        return self.decode(latent)


__all__ = ["HybridNATAutoencoder", "HybridConfig"]
