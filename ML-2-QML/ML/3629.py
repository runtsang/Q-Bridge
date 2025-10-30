"""Hybrid classical classifier that mirrors the quantum interface."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNClassifier(nn.Module):
    """CNN + FC classifier compatible with the quantum helper interface."""

    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        # Feature extractor similar to QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten size depends on input size; assume 28x28 images
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.fc(flat)
        return self.norm(logits)


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Sequential, list[int], list[int], list[int]]:
    """Return a simple feed‑forward network and metadata."""
    layers = []
    in_dim = num_features
    weight_sizes = []
    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, list(range(num_features)), weight_sizes, observables


__all__ = ["HybridCNNClassifier", "build_classifier_circuit"]
