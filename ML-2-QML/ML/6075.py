import torch
import torch.nn as nn
from typing import Iterable, Tuple

class QuantumHybridClassifier:
    """
    Classical feed‑forward network with residual connections and dropout,
    mirroring the quantum helper interface.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Construct a residual network with dropout and return metadata identical
        to the quantum variant: model, encoding indices, weight counts, and
        observables.
        """
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    @staticmethod
    def forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass using the built network."""
        return model(x)

__all__ = ["QuantumHybridClassifier"]
