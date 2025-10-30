import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifier(nn.Module):
    """Classical neural network mimicking a quantum classifier interface."""
    def __init__(self, num_features: int, depth: int = 3, hidden_ratio: float = 1.0,
                 dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            out_dim = int(num_features * hidden_ratio)
            linear = nn.Linear(in_dim, out_dim)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return an instance of the network, its encoding indices, weight sizes, and output observables."""
        net = QuantumClassifier(num_features, depth)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in net.parameters()]
        observables = [0, 1]  # placeholder for class logits
        return net, encoding, weight_sizes, observables

__all__ = ["QuantumClassifier"]
