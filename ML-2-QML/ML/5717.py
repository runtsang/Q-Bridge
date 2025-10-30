"""Hybrid classical kernel module with learnable RBF parameters and classifier construction.

This module extends the original ``QuantumKernelMethod`` by:
- Exposing a learnable kernel width (γ) as a torch parameter.
- Providing a unified ``HybridKernelModel`` class that can be used directly as a
  kernel module or wrapped into a larger learning pipeline.
- Adding a convenient ``build_classifier_circuit`` that mirrors the quantum
  counterpart, enabling seamless comparison experiments.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple, List
import numpy as np
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Classic RBF kernel ansatz with a learnable width parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm)

class HybridKernelModel(nn.Module):
    """Wrapper that exposes the kernel as a standard torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    # Expose a utility to compute a full Gram matrix
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

# Backward‑compatibility alias
Kernel = HybridKernelModel

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two sequences of tensors."""
    kernel = HybridKernelModel(gamma)
    return kernel.kernel_matrix(a, b)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum version.

    Returns:
        network: ``torch.nn.Sequential`` with ``depth`` hidden layers.
        encoding: indices of input features mapped to the network.
        weight_sizes: number of trainable parameters per layer.
        observables: dummy placeholder for API parity.
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

__all__ = ["KernalAnsatz", "HybridKernelModel", "Kernel", "kernel_matrix", "build_classifier_circuit"]
