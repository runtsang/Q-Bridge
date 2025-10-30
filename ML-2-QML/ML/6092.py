"""Hybrid classical front‑end for QuantumNAT that outputs features for quantum processing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNAT(nn.Module):
    """
    Classical feature extractor that mirrors the original QuantumNAT CNN but with
    additional regularisation and a flexible output interface.
    The returned feature tensor can be fed into a quantum module for hybrid training.
    """

    def __init__(self, n_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.  The output has shape ``(batch, n_classes)``.
        """
        x = self.features(x)
        x = self.flatten(x)
        return self.fc(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 128‑dimensional feature vector before the final classification head.
        Useful when coupling to a quantum layer.
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc[0](x)  # first linear layer
        return x


__all__ = ["HybridQuantumNAT"]
