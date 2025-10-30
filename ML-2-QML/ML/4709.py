"""Hybrid classical sampler network combining CNN feature extraction,
autoencoder latent representation, and a fully connected classifier.

The architecture is inspired by the original two‑layer MLP, the
Quantum‑NAT CNN, and the Autoencoder example.  It outputs a
softmax over four classes and exposes the latent representation
for auxiliary tasks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    Classical sampler network that mirrors the quantum SamplerQNN
    but augments the original two‑layer MLP with a CNN encoder
    (inspired by QuantumNAT) and a lightweight autoencoder
    (inspired by Autoencoder.py). The network outputs a softmax
    over four classes and also exposes the latent representation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        latent_dim: int = 16,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Feature extractor (QuantumNAT style)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Autoencoder (Autoencoder.py style)
        self.autoencoder = self._build_autoencoder(latent_dim, hidden_dims, dropout)

        # Final classifier
        # Compute flattened size after conv layers for typical 28x28 input
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat = self.features(dummy)
        flat_dim = feat.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def _build_autoencoder(
        self, latent_dim: int, hidden_dims: tuple[int, int], dropout: float
    ):
        encoder_layers = []
        in_dim = 16 * 7 * 7  # assuming 28x28 input after convs
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, 16 * 7 * 7))
        decoder = nn.Sequential(*decoder_layers)

        class Autoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.encoder(x)

            def decode(self, z: torch.Tensor) -> torch.Tensor:
                return self.decoder(z)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.decode(self.encode(x))

        return Autoencoder(encoder, decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction
        feat = self.features(x)          # (B, 16, 7, 7)
        flat = feat.view(feat.size(0), -1)
        # Autoencoder latent
        latent = self.autoencoder.encode(flat)
        # Classification
        logits = self.fc(latent)
        probs = F.softmax(logits, dim=-1)
        return self.norm(probs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation for the input."""
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        return self.autoencoder.encode(flat)


__all__ = ["SamplerQNN"]
