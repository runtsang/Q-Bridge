"""QuantumNATEnhanced: Classical CNN‑to‑FC pipeline with a learnable quantum‑inspired feature map and ensemble post‑processing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RandomFourierFeatureMap(nn.Module):
    """Random Fourier feature map to approximate a quantum kernel."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(2 * np.pi * torch.rand(output_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        return torch.cos(x @ self.W + self.b)

class QuantumNATEnhanced(nn.Module):
    """Classical model that extends the original Quantum‑NAT architecture."""
    def __init__(self, n_features: int = 4, n_circuits: int = 3) -> None:
        super().__init__()
        # Feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Projection to a feature vector
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # Ensemble of quantum‑inspired feature maps
        self.quantum_feature_maps = nn.ModuleList(
            [RandomFourierFeatureMap(32, 64) for _ in range(n_circuits)]
        )
        # Classical post‑processing
        self.postprocess = nn.Linear(n_circuits * 64, n_features)
        self.norm = nn.BatchNorm1d(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.cnn(x).view(bsz, -1)
        feat = self.fc(feat)
        # Ensemble quantum‑inspired features
        qfeat = torch.cat([m(feat) for m in self.quantum_feature_maps], dim=1)
        out = self.postprocess(qfeat)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
