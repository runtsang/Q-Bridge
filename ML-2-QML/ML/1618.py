import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier mirroring the quantum helper interface.
    Supports configurable depth, hidden size, dropout, activation, and
    multi‑class output.  Provides metadata (encoding, weight sizes,
    observables) compatible with the quantum counterpart.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 hidden_dim: int | None = None,
                 dropout: float = 0.0,
                 n_classes: int = 2,
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = hidden_dim

        head = nn.Linear(in_dim, n_classes)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    @classmethod
    def build_classifier_circuit(cls,
                                 num_features: int,
                                 depth: int = 3,
                                 hidden_dim: int | None = None,
                                 dropout: float = 0.0,
                                 n_classes: int = 2,
                                 activation: nn.Module = nn.ReLU()) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a feed‑forward classifier and return metadata identical to
        the quantum helper.

        Returns:
            network: nn.Module
            encoding: List[int]  # mapping of feature indices
            weight_sizes: List[int]  # number of parameters per layer
            observables: List[int]  # class indices
        """
        model = cls(num_features, depth, hidden_dim, dropout, n_classes, activation)
        observables = list(range(n_classes))
        return model, model.encoding, model.weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
