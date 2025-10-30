"""Classical implementation of a hybrid kernel method and classifier factory.

The class provides:
- a static method :func:`rbf_kernel_matrix` for computing an RBF kernel Gram matrix;
- a static method :func:`build_classifier_circuit` that returns a feed‑forward neural
  network and metadata in the same signature as the quantum helper.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# --------------------------------------------------------------------------- #
# RBF kernel utilities
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Purely classical RBF kernel implemented with PyTorch."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes the same API as the quantum counterpart."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernel(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Classifier factory
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a feed‑forward network and metadata mimicking the quantum helper signature.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

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

# --------------------------------------------------------------------------- #
# Shared class
# --------------------------------------------------------------------------- #

class QuantumKernelMethod:
    """
    Unified interface for kernel evaluation and classifier construction.

    The class deliberately exposes the same public API as the quantum implementation
    so that callers can instantiate either the classical or quantum version
    without changing downstream code.
    """

    @staticmethod
    def rbf_kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        return kernel_matrix(a, b, gamma)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        return build_classifier_circuit(num_features, depth)

__all__ = [
    "RBFKernel",
    "Kernel",
    "kernel_matrix",
    "build_classifier_circuit",
    "QuantumKernelMethod",
]
