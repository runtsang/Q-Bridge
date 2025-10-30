"""Hybrid classical classifier that mirrors the quantum helper interface.

The network consists of:
1. A small CNN feature extractor (Quantum‑NAT style).
2. A sequence of fully‑connected layers mimicking QCNN convolution steps.
3. A binary linear head with batch‑norm.

The factory function returns the network together with metadata that a
quantum counterpart can consume (encoding indices, weight sizes and
placeholder observables).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

class HybridClassifier(nn.Module):
    """CNN + FC stack inspired by Quantum‑NAT and QCNN."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # Stage 1 – convolutional feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Stage 2 – “quantum‑like” fully‑connected layers (QCNN style)
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 7 * 7, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
        )
        # Stage 3 – final classifier
        self.classifier = nn.Linear(4, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.classifier(x)
        return self.bn(x)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a hybrid classical network that mimics the quantum helper.

    Parameters
    ----------
    num_features : int
        Number of input features (flattened). Used to compute weight sizes.
    depth : int
        Unused but kept for API compatibility.

    Returns
    -------
    network : nn.Module
        The constructed hybrid classifier.
    encoding : Iterable[int]
        Dummy encoding indices that the quantum side can map to input features.
    weight_sizes : Iterable[int]
        Total number of trainable parameters per linear layer.
    observables : list[int]
        Placeholder list of observable indices (e.g. qubit indices) that
        the quantum side would measure.
    """
    network = HybridClassifier()
    weight_sizes = []
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    encoding = list(range(num_features))
    observables = list(range(num_features))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit", "HybridClassifier"]
