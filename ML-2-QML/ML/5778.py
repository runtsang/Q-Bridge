"""Classical kernel and classifier utilities combined into a unified interface.

The class exposes both a classical radial basis function kernel and a simple
feed‑forward neural network classifier.  The design mirrors the quantum
interface so that the same API can be swapped for a quantum implementation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence
import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of samples."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module,
                                                                   Iterable[int],
                                                                   Iterable[int],
                                                                   list[int]]:
    """Construct a simple feed‑forward classifier and return metadata."""
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

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


class QuantumKernelMethod:
    """Unified classical interface for kernel evaluation and classifier construction."""
    def __init__(self, gamma: float = 1.0, num_qubits: int = 4, depth: int = 2) -> None:
        self.gamma = gamma
        self.num_qubits = num_qubits
        self.depth = depth

    def classical_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return Kernel(self.gamma)(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, self.gamma)

    def build_classifier(self, num_features: int, depth: int) -> Tuple[nn.Module,
                                                                     Iterable[int],
                                                                     Iterable[int],
                                                                     list[int]]:
        return build_classifier_circuit(num_features, depth)

    def __repr__(self) -> str:
        return f"<QuantumKernelMethod gamma={self.gamma} num_qubits={self.num_qubits} depth={self.depth}>"

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix",
           "build_classifier_circuit", "QuantumKernelMethod"]
