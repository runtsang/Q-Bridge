"""Hybrid natural image classifier – classical implementation.

The model combines a standard convolutional backbone, a learnable
ConvFilter (mimicking a quanvolution), and a fully‑connected head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvFilter(nn.Module):
    """
    A lightweight convolutional filter that emulates the quantum
    quanvolution.  It learns a single 2×2 kernel and applies a sigmoid
    activation to produce a scalar feature for each patch.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, 1, H, W)
        logits = self.conv(x)  # (B, 1, H-k+1, W-k+1)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce to a single scalar per patch
        return activations.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)


class HybridNATModel(nn.Module):
    """
    Classical hybrid model that replaces the quantum quanvolution with
    ConvFilter while keeping the rest of the architecture identical to the
    original QFCModel.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum‑inspired filter
        self.quantum_filter = ConvFilter(kernel_size=2, threshold=0.0)
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),  # +1 for the ConvFilter scalar
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        feats = self.features(x)                     # (B, 16, 7, 7)
        # Apply ConvFilter across the whole image
        qfeat = self.quantum_filter(x)                # (B, 1, 1, 1)
        qfeat = qfeat.view(bsz, -1)                  # (B, 1)
        flat = feats.view(bsz, -1)                   # (B, 16*7*7)
        concat = torch.cat([flat, qfeat], dim=1)      # (B, 16*7*7+1)
        out = self.fc(concat)
        return self.norm(out)


__all__ = ["HybridNATModel"]
