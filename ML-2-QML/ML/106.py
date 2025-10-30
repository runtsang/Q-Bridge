import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class QuantumRBFKernel(nn.Module):
    """Classical RBF kernel with batched support and optional multiple gammas.

    The kernel is defined as
        k(x, y) = exp(-γ ||x - y||^2)
    where γ can be a scalar or a tensor broadcastable across the input dimension.
    """
    def __init__(self, gamma: float | torch.Tensor = 1.0) -> None:
        super().__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel matrix between two batches of vectors."""
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        diff = x[:, None, :] - y[None, :, :]
        sq_dist = torch.einsum('ijk,ijk->ij', diff, diff)
        return torch.exp(-self.gamma * sq_dist)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper that accepts a list of tensors and returns a NumPy array."""
        device = a[0].device
        x = torch.cat(a).to(device)
        y = torch.cat(b).to(device)
        return self.forward(x, y).detach().cpu().numpy()

__all__ = ["QuantumRBFKernel"]
