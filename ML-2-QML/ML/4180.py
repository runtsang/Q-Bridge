"""
Hybrid classical model combining convolutional feature extraction,
a small fully‑connected estimator, and QCNN‑style linear layers.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNHybridModel(nn.Module):
    """
    Classical counterpart of the hybrid QCNN model.

    Architecture:
        Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool
        → Flatten → Linear(64) → ReLU → Linear(4)
        → Linear(8) → Tanh → Linear(1)
        → BatchNorm1d
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projection (QCNN style)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # EstimatorQNN‑like small network
        self.estimator = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalized scalar output per sample.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.estimator(out)
        return self.norm(out)


__all__ = ["QCNNHybridModel"]
