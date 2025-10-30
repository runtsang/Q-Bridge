"""Hybrid QCNN-inspired model combining classical convolution, linear layers, and a depth‑controlled classifier.

The architecture mirrors the quantum QCNN pipeline but remains fully classical.
It integrates the 2×2 patch extraction from the Quanvolution example and a
depth‑parameterised feed‑forward head inspired by QuantumClassifierModel.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNGen208(nn.Module):
    """Classical QCNN generator.

    Parameters
    ----------
    num_features : int, default 8
        Dimensionality after the initial feature map.
    depth : int, default 3
        Number of linear layers in the classifier head.
    """

    def __init__(self, num_features: int = 8, depth: int = 3) -> None:
        super().__init__()
        # 2×2 patch extraction as in Quanvolution
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Flatten to a 1‑D feature vector
        self.flatten = nn.Flatten()
        # Feature map to match quantum input dimension (8)
        self.feature_map = nn.Linear(4 * 14 * 14, num_features)
        # Depth‑controlled classifier
        layers = [self.feature_map]
        for _ in range(depth):
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_features, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Input shape: (batch, 1, 28, 28)
        patches = self.patch_conv(x)
        flat = self.flatten(patches)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

    @property
    def weight_sizes(self) -> list[int]:
        """Return number of trainable parameters per layer."""
        return [p.numel() for p in self.parameters()]


def QCNNGen208Factory(num_features: int = 8, depth: int = 3) -> QCNNGen208:
    """Convenience factory mimicking the original QCNN() helper."""
    return QCNNGen208(num_features=num_features, depth=depth)


__all__ = ["QCNNGen208", "QCNNGen208Factory"]
