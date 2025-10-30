"""Hybrid classical kernel: autoencoder-based dimensionality reduction followed by RBF."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Tuple

# --------------------------------------------------------------------------- #
# Autoencoder definitions (adapted from the original seed)
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    """Configuration for the autoencoder network."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable hidden layers."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Convenience factory mirroring the original API."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Hybrid kernel: RBF applied to autoencoder latent codes
# --------------------------------------------------------------------------- #

class HybridKernel(nn.Module):
    """Hybrid RBF kernel that first compresses data with an autoencoder."""
    def __init__(self, autoencoder: nn.Module, gamma: float = 1.0) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2‑D tensors
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        z_x = self.autoencoder.encode(x)
        z_y = self.autoencoder.encode(y)
        diff = z_x - z_y
        # RBF on latent space
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  autoencoder: nn.Module,
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of tensors."""
    kernel = HybridKernel(autoencoder, gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet",
           "HybridKernel", "kernel_matrix"]
