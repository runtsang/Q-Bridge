from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
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

class QuantumClassifierModel:
    """Hybrid classifier interface with classical backbone and optional quantum extension."""
    def __init__(self, num_features: int, depth: int, use_quantum: bool = False):
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum
        self.classical_net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        return self.classical_net(x)

    def forward_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder for quantum forward pass; disabled in this classical module."""
        raise NotImplementedError("Quantum part is disabled in this classical module.")
