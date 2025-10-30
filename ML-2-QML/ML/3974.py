"""Hybrid classical classifier that mirrors the quantum interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import torch
from torch import nn, Tensor

class QuantumClassifierModel:
    """
    Classical implementation of a classifier that shares the same API as the quantum counterpart.
    - build_classifier_circuit constructs a feed‑forward neural network with the same metadata
      (encoding indices, weight sizes, output observables) as the quantum circuit.
    - build_kernel returns a radial‑basis‑function kernel, matching the quantum kernel's interface.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """Build a feed‑forward network."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding: List[int] = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            layers.extend([lin, nn.ReLU(inplace=True)])
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables

    @staticmethod
    def build_kernel(gamma: float = 1.0) -> "Kernel":
        """Return an instance of the classical RBF kernel that mimics the quantum kernel API."""
        return Kernel(gamma)

class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # pragma: no cover
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper exposing the same interface as the quantum kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # pragma: no cover
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Iterable[Tensor], b: Iterable[Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of feature vectors using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumClassifierModel", "Kernel", "kernel_matrix", "KernalAnsatz"]
