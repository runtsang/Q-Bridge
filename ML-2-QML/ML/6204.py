"""
Classical hybrid classifier that mirrors the quantum interface.

The class builds a layered feed‑forward network and optionally
appends a classical FullyConnectedLayer (stand‑in for a quantum
layer).  It exposes the same public API as the quantum variant:
```
build_classifier_circuit(...) -> (network, encoding, weight_sizes, observables)
QuantumClassifierModel(num_features, depth, use_fcl=False)
```
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List

class FullyConnectedLayer(nn.Module):
    """
    Classical stand‑in for the quantum fully‑connected layer.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

def build_classifier_circuit(num_features: int, depth: int,
                            use_fcl: bool = False) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    Args:
        num_features: dimension of the input data.
        depth: number of hidden layers.
        use_fcl: if True, append a fully‑connected layer at the end.
    Returns:
        network: nn.Sequential model.
        encoding: list of indices corresponding to input features.
        weight_sizes: list of parameter counts per layer.
        observables: list of dummy observable indices (for API parity).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

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

    if use_fcl:
        fcl = FullyConnectedLayer(n_features=1)
        network.add_module("fcl", fcl)

    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class QuantumClassifierModel(nn.Module):
    """
    Classical hybrid classifier that mirrors the quantum interface.
    """
    def __init__(self, num_features: int, depth: int, use_fcl: bool = False) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, use_fcl
        )
        self.use_fcl = use_fcl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

__all__ = ["FullyConnectedLayer", "build_classifier_circuit", "QuantumClassifierModel"]
