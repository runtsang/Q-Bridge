"""Hybrid classical model combining CNN feature extraction, a fully‑connected head, and a sampler network.

The architecture is inspired by Quantum‑NAT and SamplerQNN, providing a 4‑dimensional latent space that can be fed into the quantum module.  The `encode_for_quantum` method exposes this latent representation for downstream quantum processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNAT(nn.Module):
    """Hybrid classical model for image classification.

    The network consists of:
    * Convolutional feature extractor (3 conv layers with batch‑norm, ReLU, max‑pool, dropout).
    * Fully‑connected head producing 4 logits.
    * Sampler network that maps the same features to a 2‑dim probability vector.
    The 4‑dim output is the latent vector intended for the quantum module.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Sampler network (mirrors SamplerQNN)
        self.sampler_net = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def encode_for_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 4‑dim latent representation for the quantum module."""
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        return self.fc[0](flat)  # 128‑dim, then will be reduced in quantum module

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and sampler probabilities."""
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        logits = self.norm(self.fc[1:](flat))
        sampler_logits = self.sampler_net(flat)
        probs = F.softmax(sampler_logits, dim=-1)
        return logits, probs


__all__ = ["HybridQuantumNAT"]
