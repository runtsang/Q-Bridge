"""Hybrid classical model combining CNN feature extraction and a sampler network.

This module defines QuantumNATHybrid, a purely classical neural network that
extracts features with a small convolutional backbone, projects them to a
latent space, and then uses a small sampler network to produce a probability
distribution over four target classes.  The sampler is inspired by the
SamplerQNN architecture and allows easy replacement with a quantum
implementation if desired.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """Simple sampler that maps latent vectors to class probabilities."""
    def __init__(self, latent_dim: int = 4, n_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

class QuantumNATHybrid(nn.Module):
    """Classical hybrid architecture that mirrors the Quantum-NAT design.

    The network consists of:
    * 2-layer CNN for low‑level feature extraction.
    * Fully connected projection to a 4‑dimensional latent space.
    * SamplerModule that turns the latent vector into a probability distribution.
    """
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
        # Projection to latent space
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Sampler to produce class probabilities
        self.sampler = SamplerModule(latent_dim=4, n_classes=4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        latent = self.fc(flat)
        probs = self.sampler(latent)
        return self.norm(probs)

__all__ = ["QuantumNATHybrid"]
