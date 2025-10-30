from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridAutoencoderConfig:
    input_channels: int = 1
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(16 * 7 * 7, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.latent_dim),
        )
        self.quantum_layer = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], 16 * 7 * 7),
            nn.ReLU(),
        )
        self.reconstruction = nn.Sequential(
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, config.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        latent = self.fc_enc(flattened)
        return latent

    def quantum(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantum_layer(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(z)
        return self.reconstruction(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.quantum(z)
        return self.decode(z)

__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig"]
