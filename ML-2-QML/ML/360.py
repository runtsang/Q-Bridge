import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """Hybrid‑classical classifier with residual connections and optional dropout.
    Mirrors the quantum helper interface for seamless integration.
    """

    def __init__(self, num_features: int, depth: int = 1,
                 dropout: float = 0.0, num_classes: int = 2):
        self.num_features = int(num_features)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)
        self._build_network()

    def _build_network(self) -> None:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        self.encoding = list(range(self.num_features))
        self.weight_sizes: List[int] = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features

        head = nn.Linear(in_dim, self.num_classes)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())
        self.network = nn.Sequential(*layers)

    def build(self) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """Return the classifier, encoding indices, weight sizes, and output indices."""
        return self.network, self.encoding, self.weight_sizes, list(range(self.num_classes))

    @staticmethod
    def get_class_name() -> str:
        return "QuantumClassifierModel"
