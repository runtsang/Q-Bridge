import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Classical hybrid kernel combining RBF and polynomial components.

    The kernel is a weighted sum of an RBF kernel and a polynomial
    kernel.  The weighting parameters (gamma, degree, alpha) are
    learnable, allowing the model to discover the optimal mix for a
    downstream task.
    """
    def __init__(self, gamma: float = 1.0, degree: int = 3, alpha: float = 0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.degree = degree
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))

    def rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.dot(x, y) + 1) ** self.degree

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Weighted sum of RBF and polynomial kernels."""
        return self.alpha * self.rbf(x, y) + (1 - self.alpha) * self.poly(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
