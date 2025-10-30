import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class QuantumClassifierModel(nn.Module):
    """Hybrid classical classifier with attention and residual connections."""
    def __init__(self, num_features: int, depth: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.residual = nn.Identity()
        self.head = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attn(x)
        x = x * attn_weights
        out = self.body(x)
        out = out + self.residual(x)
        logits = self.head(out)
        return logits

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return a tuple (network, encoding, weight_sizes, observables)."""
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
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

__all__ = ["QuantumClassifierModel"]
