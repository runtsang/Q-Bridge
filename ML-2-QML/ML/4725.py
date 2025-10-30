"""Hybrid estimator combining autoencoding, convolutional feature extraction,
and quantum regression.

The architecture:
  * Classical autoencoder compresses input → latent vector.
  * QCNN module processes the latent vector.
  * Linear head outputs a regression value.

This design merges concepts from EstimatorQNN (simple feed‑forward),
QCNN (convolutional layers) and Autoencoder (dimensionality reduction)
into a single trainable PyTorch model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


# ---------- Autoencoder ----------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder."""

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


# ---------- QCNN ----------
class QCNNModel(nn.Module):
    """Convolution‑style network that can handle arbitrary input sizes."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 2 * input_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(2 * input_dim, 2 * input_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(2 * input_dim, 1.5 * input_dim), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(1.5 * input_dim, input_dim), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(input_dim, 0.5 * input_dim), nn.Tanh())
        self.head = nn.Linear(0.5 * input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return torch.sigmoid(self.head(x))


# ---------- Hybrid estimator ----------
class HybridEstimatorQNN(nn.Module):
    """Complete model: autoencoder → QCNN → linear head."""

    def __init__(self, auto_cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(auto_cfg)
        self.qcnn = QCNNModel(auto_cfg.latent_dim)
        self.final = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.autoencoder.encode(x)
        y = self.qcnn(z)
        return self.final(y)


def EstimatorQNN(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridEstimatorQNN:
    """Factory that builds a fully‑connected hybrid estimator."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridEstimatorQNN(cfg)


__all__ = [
    "EstimatorQNN",
    "HybridEstimatorQNN",
    "AutoencoderConfig",
    "AutoencoderNet",
    "QCNNModel",
]
