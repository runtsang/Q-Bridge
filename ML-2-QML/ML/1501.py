import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifier(nn.Module):
    """Classical neural network that mimics the interface of a quantum classifier.

    The network consists of `depth` hidden layers with optional batch‑normalisation
    and dropout.  It exposes the same metadata (encoding, weight sizes, observables)
    as the quantum implementation to simplify downstream integration.
    """
    def __init__(self, num_features: int, depth: int = 3, use_dropout: bool = False,
                 dropout_rate: float = 0.5, use_batchnorm: bool = False):
        super().__init__()
        self.depth = depth
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = num_features
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int = 3,
                                 use_dropout: bool = False,
                                 dropout_rate: float = 0.5,
                                 use_batchnorm: bool = False
                                ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return a model together with metadata compatible with the quantum helper."""
        model = QuantumClassifier(num_features, depth, use_dropout,
                                  dropout_rate, use_batchnorm)
        encoding = list(range(num_features))
        weight_sizes = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight_sizes.append(module.weight.numel() + module.bias.numel())
        observables = list(range(2))
        return model, encoding, weight_sizes, observables

__all__ = ["QuantumClassifier"]
