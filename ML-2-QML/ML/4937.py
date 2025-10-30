"""Hybrid quantum‑classical classifier – classical implementation.

This module implements a hybrid architecture that mimics the quantum helper
interface while leveraging classical convolutional and linear layers inspired
by the Quanvolution and QCNN examples.  It can be swapped in wherever the
original `QuantumClassifierModel.py` was used.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFeatureExtractor(nn.Module):
    """Combined quanvolution + QCNN inspired feature extractor.

    * A 2×2 patch convolution (mimicking a quanvolution filter) followed by
      a stack of linear layers that emulate the QCNN feature map.
    """
    def __init__(self) -> None:
        super().__init__()
        # 2×2 patch convolution – 4 output channels
        self.qconv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # QCNN‑style linear stack (64‑dimensional embedding)
        self.qcnn_layers = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, 1, 28, 28)
        patches = self.qconv(x)                     # (N, 4, 14, 14)
        flat = patches.view(patches.size(0), -1)    # (N, 4*14*14)
        return self.qcnn_layers(flat)


class HybridQuantumClassifier(nn.Module):
    """Classic classifier that mirrors the quantum helper interface."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.feature_extractor = HybridFeatureExtractor()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


def build_classifier_circuit(
    num_features: int = 4 * 14 * 14,
    depth: int = 3,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a classical feed‑forward network and metadata.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network with ReLU activations.
    encoding : Iterable[int]
        Dummy encoding indices, mirroring the quantum variant.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Placeholder indices for post‑measurement; kept for API parity.
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
    observables = [0, 1]  # placeholder
    return network, encoding, weight_sizes, observables


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
