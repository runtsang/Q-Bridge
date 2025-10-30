"""Hybrid Quanvolution + Autoencoder implemented entirely in PyTorch.

This module extends the original `Quanvolution` example by adding a
fully‑connected autoencoder and a dropout‑aware classification head.
The design is intentionally modular so that each block can be
experimented with independently.

The implementation imports only the standard PyTorch stack, making it
fully classical and suitable for GPU acceleration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# --------------------------------------------------------------------------- #
# 1. Autoencoder utilities – identical to the reference Autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Convenience factory mirroring the quantum helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 2. Classical quanvolution filter – expanded from the seed
# --------------------------------------------------------------------------- #
class QuanvolutionFilterClassic(nn.Module):
    """2×2 patch extractor followed by a shallow CNN."""

    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 8 output feature maps
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=2, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).flatten(1)


# --------------------------------------------------------------------------- #
# 3. Hybrid classifier – combines the filter, autoencoder and head
# --------------------------------------------------------------------------- #
class QuanvolutionAutoencoderQNN(nn.Module):
    """Classical hybrid model.

    The forward pass follows:
        1. 2×2 patch extraction via a shallow conv.
        2. Flattened features → MLP autoencoder (encoder only).
        3. Linear classifier on the latent code.
    """

    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterClassic()
        # The flattened feature dimension is 8 * 14 * 14
        self.autoencoder = Autoencoder(
            input_dim=8 * 14 * 14,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)          # shape: [N, 8*14*14]
        latent = self.autoencoder.encode(features)
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Autoencoder", "AutoencoderNet", "AutoencoderConfig", "QuanvolutionFilterClassic", "QuanvolutionAutoencoderQNN"]
