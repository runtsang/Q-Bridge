"""Hybrid classical model for Quantum‑NAT with an extended architecture.

The new `QuantumNAT__gen503` class builds on the original CNN‑FC pipeline but adds:
* A learnable embedding that is concatenated with CNN features.
* A small parameterized MLP that mimics the quantum circuit.
* A final linear head producing the 4‑dimensional output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class _CNNFeatureExtractor(nn.Module):
    """Lightweight CNN that mirrors the original architecture but allows more channels."""
    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # output shape (bs, out_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.shape[0], -1)  # (bs, out_channels)

class _ClassicalFusion(nn.Module):
    """Concatenates CNN features with a learnable embedding."""
    def __init__(self, feature_dim: int, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(embed_dim))
        self.embed_dim = embed_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz = features.shape[0]
        embed = self.embed.expand(bsz, -1)  # broadcast to batch
        return torch.cat([features, embed], dim=1)

class _ClassicalQuantumLayer(nn.Module):
    """A small MLP that emulates the quantum circuit output."""
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumNAT__gen503(nn.Module):
    """Hybrid classical approximation of the Quantum‑NAT model."""
    def __init__(self) -> None:
        super().__init__()
        self.cnn = _CNNFeatureExtractor()
        self.fusion = _ClassicalFusion(feature_dim=32, embed_dim=8)
        self.quantum_layer = _ClassicalQuantumLayer(input_dim=40, output_dim=4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                 # (bs, 32)
        fused = self.fusion(features)          # (bs, 40)
        out = self.quantum_layer(fused)        # (bs, 4)
        return self.norm(out)

__all__ = ["QuantumNAT__gen503"]
