from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.functional import tanh


class ClassicalKernelAnsatz(nn.Module):
    """RBF kernel ansatz with a learnable gamma parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalKernel(nn.Module):
    """Wrapper that evaluates the RBF kernel for two vectors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def classical_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix using the classical RBF kernel."""
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class FullyConnectedLayer(nn.Module):
    """A minimal linear layer with tanh activation, used to fuse kernel outputs."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Sequence[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridKernel(nn.Module):
    """
    Hybrid kernel that optionally combines a classical RBF kernel with a quantum kernel.
    A lightweight fully connected layer learns the weighting between the two components.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        quantum_module: Optional[nn.Module] = None,
        combine_weights: bool = True,
    ) -> None:
        super().__init__()
        self.classical = ClassicalKernel(gamma)
        self.use_quantum = use_quantum
        self.quantum_module = quantum_module
        self.combine_weights = combine_weights
        if combine_weights:
            # Weight vector for classical and quantum contributions
            self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
            self.fcl = FullyConnectedLayer(2)
        else:
            self.fcl = FullyConnectedLayer(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_c = self.classical(x, y)
        if self.use_quantum and self.quantum_module is not None:
            k_q = self.quantum_module(x, y)
        else:
            k_q = torch.tensor(0.0, dtype=torch.float32, device=x.device)

        if self.combine_weights:
            combined = torch.stack([k_c, k_q], dim=0)
            fused = self.fcl(combined)
            return fused
        else:
            return k_c + k_q


def hybrid_kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    use_quantum: bool = False,
    quantum_module: Optional[nn.Module] = None,
    combine_weights: bool = True,
) -> np.ndarray:
    """
    Compute the Gram matrix for a dataset using the HybridKernel.
    The function mirrors the behaviour of the class for convenience.
    """
    kernel = HybridKernel(gamma, use_quantum, quantum_module, combine_weights)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "ClassicalKernelAnsatz",
    "ClassicalKernel",
    "classical_kernel_matrix",
    "FullyConnectedLayer",
    "HybridKernel",
    "hybrid_kernel_matrix",
]
