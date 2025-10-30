"""Hybrid classical model that mimics the quantum patch filter and provides metadata for a quantum counterpart."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a simple feed‑forward network and metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
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


class HybridQuanvolutionClassifier(nn.Module):
    """Classical hybrid model that mimics the quantum patch filter and provides metadata for a quantum counterpart."""

    def __init__(self, num_classes: int = 10, depth: int = 2) -> None:
        super().__init__()
        # Convolutional patch extractor
        self.patch_extractor = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        nn.init.xavier_uniform_(self.patch_extractor.weight)
        # Flatten and linear head
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )
        # Metadata for quantum counterpart
        self.classifier_network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features=4, depth=depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(x)
        features = patches.view(x.size(0), -1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)
