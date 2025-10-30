import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SamplerQNN(nn.Module):
    """Classical sampler network mirroring the quantum SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class HybridKernelMethod:
    """Hybrid kernel that can compute classical RBF, quantum, or a weighted mixture."""
    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        quantum_kernel: callable | None = None,
        weight: float = 0.5,
    ) -> None:
        self.rbf = RBFKernel(gamma)
        self.use_quantum = use_quantum
        self.quantum_kernel = quantum_kernel
        self.weight = weight
        self.sampler = SamplerQNN()

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quantum and self.quantum_kernel is not None:
            return self.weight * self.quantum_kernel(x, y) + (1 - self.weight) * self.rbf(x, y)
        return self.rbf(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sampler(inputs)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Legacy compatibility wrapper that returns a classical RBF Gram matrix."""
    hk = HybridKernelMethod(gamma=gamma)
    return hk.kernel_matrix(a, b)

__all__ = ["RBFKernel", "SamplerQNN", "HybridKernelMethod", "kernel_matrix"]
