from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from torch import nn

__all__ = ["build_classifier_circuit"]

class HybridQuantumClassifier(nn.Module):
    """Residual‑style fully‑connected network inspired by a QCNN."""

    def __init__(self, num_features: int, depth: int = 2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

        # Record weight sizes for metadata
        self.weight_sizes = [
            sum(p.numel() for p in layer.parameters())
            for layer in [
                self.feature_map, self.conv1, self.pool1,
                self.conv2, self.pool2, self.conv3, self.head
            ]
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Factory that returns a HybridQuantumClassifier, an encoding list,
    weight sizes list and a simple observable list that matches
    the quantum interface.
    """
    model = HybridQuantumClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = model.weight_sizes
    observables = [0, 1]  # placeholder observables
    return model, encoding, weight_sizes, observables
