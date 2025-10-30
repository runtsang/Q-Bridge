"""Hybrid classical classifier mirroring quantum interface."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class HybridClassifierML(nn.Module):
    """Simple CNN+FC classifier with metadata compatible with quantum interface."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # Feature extractor – 2 conv layers + pooling
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and projection to num_classes
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a classical network mimicking the quantum signature.
    Parameters
    ----------
    num_features : int
        Number of input features (flattened image size).
    depth : int
        Depth of the feature extractor (unused but kept for compatibility).
    Returns
    -------
    network : nn.Module
        The constructed network.
    encoding : Iterable[int]
        Dummy encoding mapping each input feature to a weight index.
    weight_sizes : Iterable[int]
        Sizes of each linear layer.
    observables : List[int]
        Class indices (0, 1, …, num_classes-1).
    """
    network = HybridClassifierML(in_channels=1, num_classes=2)
    # Build dummy encoding: map each input pixel to its own weight
    encoding = list(range(num_features))
    # Compute weight sizes from network
    weight_sizes = []
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifierML", "build_classifier_circuit"]
