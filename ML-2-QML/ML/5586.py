"""Hybrid quantum‑inspired CNN classifier with a classical surrogate.

The module defines a classical network that emulates the behaviour of the quantum
classifier and a helper that constructs a classical surrogate.  The architecture
combines a quanvolution‑style convolution, a QFC‑style fully connected block,
and a QCNN‑style head, mirroring the quantum pipeline while remaining purely
classical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a classical surrogate for the quantum classifier.
    The network mirrors the quantum encoding, variational layers, and measurement observables.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridCNNClassifier(nn.Module):
    """
    Classical hybrid model that emulates the quantum quanvolution + QFC + QCNN architecture.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Classical quanvolution-inspired convolution
        self.qconv = nn.Conv2d(in_channels, 8, kernel_size=2, stride=2, padding=0)
        # Classical QFC-like fully connected block
        self.fc_block = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and feed into linear layers mimicking QCNN
        self.qcnn_head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qconv(x)
        x = self.fc_block(x)
        x = x.view(x.size(0), -1)
        logits = self.qcnn_head(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["build_classifier_circuit", "HybridCNNClassifier"]
