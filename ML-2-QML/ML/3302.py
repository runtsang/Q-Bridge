"""Hybrid fully connected layer + classifier implemented in PyTorch."""

from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
from torch import nn

class HybridFCLClassifier(nn.Module):
    """
    Classical counterpart of the quantum hybrid fully‑connected classifier.

    The network mirrors the quantum ansatz:
      * `encoding` indices identify the input features.
      * `weight_sizes` reports the number of trainable parameters per layer.
      * `observables` are dummy placeholders matching the quantum version.
    """
    def __init__(self, n_features: int = 1, depth: int = 1, n_qubits: int = 1) -> None:
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.n_qubits = n_qubits

        layers: List[nn.Module] = []
        in_dim = n_features
        self.encoding = list(range(n_features))
        self.weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, n_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = n_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables = list(range(2))  # placeholder to align with quantum side

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Run a forward pass with a list/tuple of parameters."""
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Compatibility wrapper used by the original FCL interface."""
        return self.forward(thetas)

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding, weight sizes and observables indices."""
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["HybridFCLClassifier"]
