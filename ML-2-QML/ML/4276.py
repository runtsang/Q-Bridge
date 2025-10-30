"""Hybrid classical and quantum-inspired neural network.

This module defines :class:`HybridQuanvolutionAutoencoder` which
combines a classical convolutional front‑end, a fully connected projection,
and a lightweight autoencoder.  The design is loosely inspired by the
quanvolution filter, the Quantum‑NAT fully‑connected layer, and the
classical autoencoder from the provided seed projects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    """A minimal fully‑connected autoencoder used as a feature compressor."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] | list[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # encoder
        enc_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class HybridQuanvolutionAutoencoder(nn.Module):
    """Classical hybrid network that mimics a quanvolution filter, a
    fully‑connected projection, and an autoencoder for feature compression.
    """

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 32,
        autoencoder_hidden: tuple[int, int] | list[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Classical convolutional front‑end (quanvolution style)
        self.conv = nn.Conv2d(1, 8, kernel_size=2, stride=2)

        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(8 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

        # Autoencoder used as a compressor
        self.autoencoder = SimpleAutoencoder(
            input_dim=4,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional front‑end
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # Fully‑connected projection
        x = self.fc(x)

        # Autoencoder compression
        x = self.autoencoder.encode(x)

        # Classification
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionAutoencoder", "SimpleAutoencoder"]
