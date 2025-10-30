"""Classical RBF kernel with learnable hyperparameters."""
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class HybridKernel(nn.Module):
    """Classical RBF kernel with optional learnable gamma."""
    def __init__(self, gamma: float = 1.0, learnable_gamma: bool = True):
        super().__init__()
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two batches of samples."""
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (N, M, D)
        sqdist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sqdist)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, learnable_gamma: bool = False) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors."""
    kernel = HybridKernel(gamma, learnable_gamma)
    mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            mat[i, j] = kernel(xi.unsqueeze(0), yj.unsqueeze(0))
    return mat.detach().cpu().numpy()

__all__ = ["HybridKernel", "kernel_matrix"]
