"""Hybrid binary classifier with classical CNN backbone, fully‑connected head,
batch normalization, and a lightweight regression shift.

This module fuses the classical architecture from the original seed
and the fully‑connected projection of Quantum‑NAT, while adding a
small regressor that learns a scalar shift for the final sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """Classical CNN + fully‑connected + batch‑norm + shift‑regressor."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – identical to the original QCNet backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
        )
        # Batch‑norm on the 4‑dimensional feature vector (Quantum‑NAT style)
        self.norm = nn.BatchNorm1d(4)
        # Small regressor that produces a scalar shift for the classifier
        self.regressor = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        # Two‑logit classifier
        self.classifier = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a probability vector of shape (B,2)."""
        feats = self.features(x)
        feats = self.norm(feats)
        shift = self.regressor(feats).squeeze(-1)   # (B,)
        logits = self.classifier(feats) + shift.unsqueeze(-1)
        probs = self.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
