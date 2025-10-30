"""Hybrid classical model that combines a convolutional feature extractor with a
feed‑forward network mirroring a variational quantum circuit. The architecture
is intentionally symmetrical to the quantum counterpart so that hyper‑parameter
sweeps and scaling studies can be performed on both sides in a unified manner."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class QuanvolutionHybridClassifier(nn.Module):
    """Classical counterpart of the hybrid model."""

    def __init__(self, num_features: int = 4, depth: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        # Feature extractor: 2×2 kernel, stride 2, 1 input channel
        self.conv = nn.Conv2d(1, num_features, kernel_size=2, stride=2, bias=False)
        # Feed‑forward network that mirrors the quantum ansatz
        layers: List[nn.Module] = []
        in_dim = num_features * 14 * 14  # 28×28 image → 14×14 patches
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, Iterable[int], List[int], List[int]]:
        """Return a classical network that has the same layer counts and weight shapes
        as the quantum ansatz.  The function also returns metadata that can be used
        to initialise a quantum circuit with matching encoding/variational parameters."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU(inplace=True)])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

__all__ = ["QuanvolutionHybridClassifier"]
