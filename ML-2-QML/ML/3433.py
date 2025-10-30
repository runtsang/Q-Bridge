"""Hybrid classical classifier mirroring the quantum interface."""

from __future__ import annotations

import torch
import torch.nn as nn

class HybridQuantumClassifier(nn.Module):
    """A classical CNN + FC classifier inspired by Quantum‑NAT and incremental quantum data‑uploading."""
    def __init__(self, depth: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        # Feature extractor: two conv layers + pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened dim after two 2x2 poolings on 28x28 input -> 7x7
        in_features = 16 * 7 * 7
        # Classifier head
        layers: list[nn.Module] = []
        in_dim = in_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_features))
            layers.append(nn.ReLU())
            in_dim = in_features
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.classifier(flattened)
        return self.norm(logits)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
        """
        Construct a feed‑forward network that mimics the quantum ansatz structure.
        The function returns the network, an encoding mapping (identity), weight_sizes
        for each linear layer, and a dummy observable list.
        """
        layers: list[nn.Module] = []
        in_dim = num_features
        weight_sizes: list[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        net = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = list(range(2))
        return net, encoding, weight_sizes, observables
