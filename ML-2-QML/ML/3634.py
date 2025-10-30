"""Hybrid sampler‑classifier for classical experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable

class HybridSamplerClassifier(nn.Module):
    """A two‑stage network: a sampler producing a categorical distribution
    followed by a classifier operating on the sampled one‑hot vector.
    This mirrors the quantum architecture in build_classifier_circuit
    while remaining fully classical."""
    def __init__(self, num_features: int = 2, depth: int = 1) -> None:
        super().__init__()
        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        # Classifier network
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.  The sampler output is used as a softonehot
        feeding the classifier."""
        probs = F.softmax(self.sampler(x), dim=-1)
        # Convert to one‑hot via argmax for deterministic classification
        one_hot = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=-1, keepdim=True), 1.0)
        logits = self.classifier(one_hot)
        return logits

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Return a classical analogue of the quantum classifier circuit.
        The returned tuple matches the signature of the quantum helper."""
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

__all__ = ["HybridSamplerClassifier"]
