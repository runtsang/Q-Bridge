"""Enhanced classical CNN for QuantumNAT with dropout and multi‑class head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    A deeper CNN with dropout and a multi‑class classification head.
    Designed to be a drop‑in replacement for the original QFCModel.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        # Feature extractor: two conv blocks with BN, ReLU, MaxPool, and dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        # Compute the flattened feature size: input 28x28 -> after two 2x2 pools -> 7x7
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        return self.norm(logits)

__all__ = ["QuantumNATEnhanced"]
