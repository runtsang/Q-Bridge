"""Hybrid classical model combining CNN feature extractor, quantum‑inspired classifier, and feature pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple

class HybridNATModel(nn.Module):
    """Classical network that mimics a quantum circuit by using parametrized layers for classification."""
    def __init__(self, num_qubits: int = 4, depth: int = 2):
        super().__init__()
        # Feature extractor identical to QFCModel from QuantumNAT.py
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected to reduce dimensionality
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_qubits)
        )
        # Quantum‑inspired classifier built with classical layers
        self.classifier, self.encoding, self.weight_sizes, self.observables = \
            build_classifier_circuit(num_qubits, depth)
        self.norm = nn.BatchNorm1d(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        proj = self.fc(flat)
        # Encode features as inputs to classifier (mimicking quantum encoding)
        encoded = proj  # treat as classical encoding
        out = self.classifier(encoded)
        return self.norm(out)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum variant."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridNATModel", "build_classifier_circuit"]
