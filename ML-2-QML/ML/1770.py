import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from typing import Sequence

class RBFKernel(nn.Module):
    """Differentiable RBF kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = Parameter(torch.tensor(gamma, dtype=torch.float32))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

class QuantizedSimilarity(nn.Module):
    """Simple parameterized similarity that mimics a quantum overlap."""
    def __init__(self, dim: int):
        super().__init__()
        # Parameters that will be learned to shape the kernel
        self.params = Parameter(torch.randn(dim))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        # Use params to modulate similarity; acts like a learned bandwidth
        mod = torch.exp(-torch.sum((self.params - diff)**2))
        return mod

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel combining classical RBF and a parameterised quantum‑style similarity."""
    def __init__(self, dim: int, gamma: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.quantum = QuantizedSimilarity(dim)
        self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        classical = self.rbf(x, y)
        quantum = self.quantum(x, y)
        return self.alpha * classical + (1 - self.alpha) * quantum
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        kernel = torch.zeros((len(a), len(b)))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                kernel[i, j] = self.forward(xi, yj)
        return kernel.cpu().numpy()

__all__ = ["RBFKernel", "QuantizedSimilarity", "QuantumKernelMethod"]
