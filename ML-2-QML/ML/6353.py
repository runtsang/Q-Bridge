"""Hybrid classical model combining convolutional feature extraction with a sampler network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFCModel(nn.Module):
    """Convolutional backbone followed by a small sampler MLP producing a 4‑class probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Projection to feature vector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        # Sampler MLP that refines the raw logits into a probability distribution
        self.sampler = nn.Sequential(
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        # Softmax via sampler network to obtain probabilities
        probs = F.softmax(self.sampler(logits), dim=-1)
        return self.norm(probs)


__all__ = ["QFCModel"]
