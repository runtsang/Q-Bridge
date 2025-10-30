"""Hybrid classical classifier with autoencoder and QCNN inspired architecture."""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
# QCNN‑style classifier
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """Combines an autoencoder, a QCNN‑style extractor, and a linear head."""
    def __init__(self,
                 input_dim: int,
                 encoder_cfg: AutoencoderConfig | None = None,
                 classifier_depth: int = 2) -> None:
        super().__init__()
        self.encoder_cfg = encoder_cfg or AutoencoderConfig(input_dim)
        self.autoencoder = AutoencoderNet(self.encoder_cfg)
        self.classifier = QCNNModel(self.encoder_cfg.latent_dim)
        self.output = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        embed = self.classifier(z)
        logits = self.output(embed)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).long()
