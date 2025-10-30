import numpy as np
import torch
from torch import nn
from typing import Sequence

class RBFKernel(nn.Module):
    """Vectorized RBF kernel that works with single or batched inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are 2‑D: (n, d) and (m, d)
        x = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        y = y.reshape(-1, y.shape[-1]) if y.ndim > 2 else y
        diff = x[:, None, :] - y[None, :, :]
        sq_dist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_dist)

class Kernel(nn.Module):
    """Thin wrapper around :class:`RBFKernel` that preserves the original API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1 and y.ndim == 1:
            return self.kernel(x.unsqueeze(0), y.unsqueeze(0))[0, 0]
        return self.kernel(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["RBFKernel", "Kernel", "kernel_matrix"]
