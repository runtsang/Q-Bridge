"""Hybrid classical backbone for Quantum‑NAT.

This module implements a convolutional feature extractor, a
classical “quanvolution” filter, and a lightweight fully‑connected
autoencoder.  The resulting 4‑dimensional latent vector is intended
to be consumed by the quantum module defined in the QML file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Classical quanvolution filter – a 2×2 convolution with a threshold
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Drop‑in replacement for a quantum quanvolution filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # 1 input → 1 output channel
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Filtered image of shape (B, 1, H‑k+1, W‑k+1) with sigmoid activation.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


# --------------------------------------------------------------------------- #
# Simple fully‑connected autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Encoder–decoder MLP with configurable latent dimensionality."""

    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dims: tuple[int,...] = (32, 16)) -> None:
        super().__init__()
        in_dim = input_dim
        encoder_layers = []
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class QuantumNATHybrid(nn.Module):
    """Classical pipeline that produces a 4‑D latent vector for the quantum module."""

    def __init__(self) -> None:
        super().__init__()
        # 1→8→16 feature extractor (same as seed)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quanvolution filter
        self.filter = ConvFilter(kernel_size=2, threshold=0.5)
        # Autoencoder that compresses to 4‑D latent
        # Input dim = 16*7*7 = 784
        self.autoencoder = AutoencoderNet(input_dim=784, latent_dim=4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised 4‑D latent vector ready for the quantum module.
        """
        # 1. Classical quanvolution
        x = self.filter(x)
        # 2. Feature extraction
        feats = self.features(x)          # (B, 16, 7, 7)
        # 3. Flatten for autoencoder
        flat = feats.view(feats.size(0), -1)  # (B, 784)
        # 4. Encode to latent
        latent = self.autoencoder.encode(flat)  # (B, 4)
        # 5. Normalise
        return self.norm(latent)


__all__ = ["QuantumNATHybrid", "ConvFilter", "AutoencoderNet"]
