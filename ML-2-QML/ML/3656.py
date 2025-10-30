from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class HybridClassifier(nn.Module):
    """
    A flexible neural network that can act as a simple feed‑forward classifier
    or emulate a convolution‑like architecture by stacking
    fully‑connected layers with non‑linearities.
    """

    def __init__(self, num_features: int, depth: int = 2, conv_mode: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features

        if conv_mode:
            # emulate a single convolutional block: linear -> tanh -> linear -> tanh
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, num_features))
                layers.append(nn.Tanh())
                in_dim = num_features
        else:
            # plain dense stack
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, num_features))
                layers.append(nn.ReLU())
                in_dim = num_features

        layers.append(nn.Linear(in_dim, 2))  # binary output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_classifier_circuit(num_features: int, depth: int, conv_mode: bool = False) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a feed‑forward or convolution‑like classifier and expose
    metadata that mirrors the quantum helper interface.

    Returns:
        model: the instantiated :class:`HybridClassifier`
        encoding: list of feature indices that are encoded
        weight_sizes: number of trainable parameters per layer
        observables: placeholder list used by the quantum wrapper
    """
    model = HybridClassifier(num_features, depth, conv_mode)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
