"""Classical neural network classifier with extended architecture and metadata extraction."""

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List


class QuantumClassifierModel(nn.Module):
    """
    Feed‑forward classifier with optional residual connections, dropout and
    layer‑wise weight statistics. The interface mirrors the quantum helper
    so that the same class name can be used in hybrid experiments.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        dropout: float = 0.1,
        residual: bool = False
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.residual = residual

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        if self.residual:
            out = out + x
        return self.head(out)

    def weight_metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """
        Return (encoding_indices, weight_sizes, observables) analogous to the
        quantum implementation. Encoding indices are simply the input feature
        positions; weight_sizes are the number of trainable parameters per layer;
        observables are placeholder integers for consistency.
        """
        encoding = list(range(self.num_features))
        weight_sizes: List[int] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_sizes.append(m.weight.numel() + m.bias.numel())
        observables = list(range(2))
        return encoding, weight_sizes, observables

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int
    ) -> Tuple["QuantumClassifierModel", Iterable[int], Iterable[int], List[int]]:
        """
        Convenience factory that returns an instance and its metadata, matching
        the signature of the quantum build_classifier_circuit.
        """
        model = QuantumClassifierModel(num_features, depth)
        encoding, weight_sizes, observables = model.weight_metadata()
        return model, encoding, weight_sizes, observables
