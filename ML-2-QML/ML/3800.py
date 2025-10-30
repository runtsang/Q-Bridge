"""
HybridAutoQCNet – classical implementation with an optional sigmoid head.

This module keeps all classical machinery (convolution, autoencoder, and head) in pure PyTorch.
A separate quantum module can swap the head by subclassing this class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Autoencoder configuration and network
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the fully‑connected autoencoder."""
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (16, 8)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Classical head – linear layer + sigmoid
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Dense head that mimics the quantum expectation layer."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.linear(x))

# --------------------------------------------------------------------------- #
# Full hybrid‑convolutional‑autoencoder network
# --------------------------------------------------------------------------- #
class HybridAutoQCNet(nn.Module):
    """
    Convolutional backbone ➜ dense layers ➜ autoencoder ➜ classical head.

    The architecture mirrors the original QCNet but replaces the quantum
    expectation head with a differentiable sigmoid head. This form is
    useful for quick experiments or as a baseline when a quantum device
    is unavailable.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Dense layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)  # output that feeds the autoencoder

        # Autoencoder
        ae_cfg = AutoencoderConfig(
            input_dim=32,
            latent_dim=8,
            hidden_dims=(16, 8),
            dropout=0.1
        )
        self.autoencoder = AutoencoderNet(ae_cfg)

        # Classical head
        self.head = HybridHead(in_features=ae_cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        # Flatten and dense layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape: (batch, 32)

        # Autoencoder latent representation
        latent = self.autoencoder.encode(x)  # shape: (batch, 8)

        # Classical head output
        prob = self.head(latent)  # shape: (batch, 1)

        # Return a two‑class probability distribution
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["AutoencoderConfig", "AutoencoderNet", "HybridHead", "HybridAutoQCNet"]
