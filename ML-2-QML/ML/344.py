import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with learnable hyper‑parameters and optional quantum‑kernel blending."""
    def __init__(self, gamma: float = 1.0, weight: float = 0.5, use_quantum: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.weight = nn.Parameter(torch.tensor(weight))
        self.use_quantum = use_quantum
        self.quantum_kernel = None

    def set_quantum_kernel(self, quantum_kernel):
        self.quantum_kernel = quantum_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        if self.use_quantum and self.quantum_kernel is not None:
            qk = self.quantum_kernel(x, y)
            return self.weight * rbf + (1.0 - self.weight) * qk
        return rbf

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorised Gram matrix computation."""
        a = a.unsqueeze(1)  # (n_a, 1, d)
        b = b.unsqueeze(0)  # (1, n_b, d)
        diff = a - b
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Utility that returns a NumPy Gram matrix for a list of tensors."""
    ml = QuantumKernelMethod(gamma)
    mat = ml.kernel_matrix(torch.stack(a), torch.stack(b))
    return mat.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
