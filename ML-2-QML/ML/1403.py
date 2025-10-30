"""QuantumClassifierModel: classical feed‑forward network with residual connections and batch normalization.

The class mirrors the quantum helper interface by exposing a build_classifier_circuit method that
returns a PyTorch nn.Sequential model, an encoding list, weight sizes and dummy observables.
"""

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Classical classifier with residual connections and batch normalization.
    Designed to be used interchangeably with the quantum counterpart.
    """
    def __init__(self, num_features: int, depth: int):
        """
        Parameters
        ----------
        num_features: int
            Dimensionality of input features.
        depth: int
            Number of layers in the residual blocks.
        """
        self.num_features = num_features
        self.depth = depth
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(nn.BatchNorm1d(self.num_features))
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            # Residual connection
            layers.append(nn.Identity())
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """
        Mimic the quantum helper interface. Returns the network, encoding indices,
        weight sizes and dummy observables.
        """
        model = QuantumClassifierModel(num_features, depth)
        encoding = list(range(num_features))
        weight_sizes = []
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                weight_sizes.append(layer.weight.numel() + layer.bias.numel())
        observables = list(range(2))
        return model.network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
