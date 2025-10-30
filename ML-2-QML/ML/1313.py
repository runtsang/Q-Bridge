"""Hybrid classical‑quantum model for Quantum‑NAT with deeper feature extraction and variational depth control.

The design keeps the original 4‑qubit output but the input is now processed by a multi‑scale
convolutional backbone, followed by a fully‑connected head that projects to four features.
Batch‑normalisation is applied to the final logits. The class can be instantiated with
different depths of the quantum encoder for ablation studies, but the classical branch
is fully independent and can be used as a baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModelExtended(nn.Module):
    """Classical CNN + FC head that mirrors the original QFCModel but with a deeper
    multi‑scale feature extractor and optional dropout. The output dimension is kept
    at four to match the quantum counterpart.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        # Multi‑scale convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute the flattened feature size: input 28x28 -> after 3 pools -> 3x3
        self._feature_dim = 64 * 3 * 3
        self.fc = nn.Sequential(
            nn.Linear(self._feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        return self.norm(logits)

__all__ = ["QFCModelExtended"]
