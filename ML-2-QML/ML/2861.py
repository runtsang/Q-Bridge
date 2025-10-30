"""Hybrid classical autoencoder with quanvolution feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

class QuanvolutionFilter(nn.Module):
    """Convolutional filter that extracts 2×2 patches and flattens them."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

@dataclass
class AutoencoderConfig:
    """Configuration for the fully‑connected autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Multi‑layer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QuanvolutionAutoencoder(nn.Module):
    """Hybrid model: quanvolution feature extraction + classical autoencoder."""
    def __init__(self, conv_out: int = 4, latent_dim: int = 32) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(out_channels=conv_out)
        # The flattened feature size is conv_out × 14 × 14 for 28×28 MNIST
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(input_dim=conv_out * 14 * 14, latent_dim=latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed image."""
        features = self.qfilter(x)
        reconstruction = self.autoencoder(features)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        features = self.qfilter(x)
        return self.autoencoder.encode(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.autoencoder.decode(z)

__all__ = ["QuanvolutionFilter", "AutoencoderConfig", "AutoencoderNet", "QuanvolutionAutoencoder"]
