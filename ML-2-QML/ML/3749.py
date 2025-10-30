"""Hybrid classical classifier mirroring the quantum architecture.

The network emulates the data‑uploading variational ansatz from the
Quantum‑NAT example by first extracting spatial features with a 2‑D
convolutional backbone and then applying a stack of fully‑connected
ReLU layers that corresponds to the depth‑controlled quantum circuit.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a classical feed‑forward network that mirrors the
    variational circuit defined in the quantum counterpart.

    Parameters
    ----------
    num_features: int
        Width of each hidden layer.
    depth: int
        Number of hidden layers.

    Returns
    -------
    network: nn.Sequential
        The classifier network.
    encoding: List[int]
        Dummy list representing the encoding indices (kept for API
        compatibility with the quantum side).
    weight_sizes: List[int]
        Number of trainable parameters in each layer.
    observables: List[int]
        Dummy list representing measurement observables.
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []

    # Hidden layers
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Output layer
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]  # placeholder
    return network, encoding, weight_sizes, observables


class HybridQuantumClassifier(nn.Module):
    """
    Classical counterpart to the quantum data‑uploading classifier.

    The architecture consists of:
      * A 2‑D convolutional feature extractor (inspired by Quantum‑NAT).
      * A fully‑connected projection to 4 latent features.
      * A depth‑controlled classifier network that emulates the
        variational quantum ansatz.
    """

    def __init__(self, num_features: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, num_features))
        self.norm = nn.BatchNorm1d(num_features)

        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        latent = self.fc(flat)
        latent = self.norm(latent)
        logits = self.classifier(latent)
        return logits


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
