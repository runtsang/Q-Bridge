"""Hybrid classical model combining CNN feature extraction and a depth‑parameterized classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

def build_classifier_circuit(num_features: int, depth: int) -> nn.Module:
    """
    Construct a feed‑forward classifier mirroring the quantum helper interface.
    The network consists of `depth` hidden layers of size `num_features`,
    each followed by ReLU, and a final linear layer to 4 classes.
    """
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 4))
    return nn.Sequential(*layers)

class HybridNATModel(nn.Module):
    """
    Classical CNN followed by a configurable depth‑parameterized classifier.
    The architecture is inspired by the Quantum‑NAT paper and the
    incremental classifier factory.
    """
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = build_classifier_circuit(16 * 7 * 7, depth)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.classifier(flat)
        return self.norm(logits)

__all__ = ["HybridNATModel"]
