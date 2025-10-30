"""Enhanced hybrid model combining CNN, autoencoder, and quantum-inspired projections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """Classical CNN + autoencoder pipeline producing 4‑dimensional embeddings."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected encoder producing latent vector
        self.encoder = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
        )
        # Decoder (optional reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 16 * 7 * 7),
            nn.ReLU(),
        )
        # Projection to final 4‑dimensional output
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation of input."""
        feat = self.features(x)
        flat = feat.view(feat.shape[0], -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct feature map from latent vector."""
        flat = self.decoder(z)
        return flat.view(-1, 16, 7, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        out = self.fc(z)
        return self.norm(out)

__all__ = ["QuantumNATHybrid"]
