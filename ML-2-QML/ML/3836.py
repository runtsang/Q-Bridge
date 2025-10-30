"""Hybrid natural‑language and regression model combining CNN features with a quantum‑inspired random projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


# ------------------------------------------- #
#   Data generation utilities
# ------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data that mimics a quantum superposition:
        |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    The target is a smooth function of θ and φ.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and a scalar target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ------------------------------------------- #
#   Quantum‑inspired projection layer
# ------------------------------------------- #
class RandomProjection(nn.Module):
    """
    A fixed random linear transformation that mimics the effect of a random quantum layer.
    The projection matrix is orthonormalised to emulate unitary evolution.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        w = torch.randn(out_features, in_features)
        # QR factorisation to obtain an orthonormal matrix
        q, _ = torch.linalg.qr(w.T)
        self.register_buffer("proj", q.T[:out_features])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.proj)


# ------------------------------------------- #
#   Hybrid model
# ------------------------------------------- #
class HybridNATRegression(nn.Module):
    """
    A two‑head neural network that mirrors the original Quantum‑NAT architecture
    but adds a regression head. The CNN extracts spatial features, a random
    projection substitutes the quantum layer, and two separate fully‑connected
    heads predict class logits (4‑way) and a continuous target.
    """
    def __init__(self, in_channels: int = 1, num_features: int = 16):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        flat_dim = 32  # after pooling

        # Quantum‑inspired projection
        self.proj = RandomProjection(flat_dim, 64)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.BatchNorm1d(4),
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [batch, 4] for classification
            pred:   [batch]  for regression
        """
        bsz = x.shape[0]
        feats = self.features(x).view(bsz, -1)
        proj_feats = self.proj(feats)
        logits = self.cls_head(proj_feats)
        pred = self.reg_head(proj_feats).squeeze(-1)
        return logits, pred


__all__ = ["HybridNATRegression", "RegressionDataset", "generate_superposition_data"]
