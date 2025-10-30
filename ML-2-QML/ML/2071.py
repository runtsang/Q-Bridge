import numpy as np
import torch
from torch import nn
from typing import Sequence, Union

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with optional anisotropic gamma and vectorized computation.
    The class can be used directly in PyTorch pipelines and exposes a convenient
    `kernel_matrix` method.
    """
    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0):
        super().__init__()
        if isinstance(gamma, (list, tuple)):
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
        else:
            self.gamma = torch.tensor([gamma], dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel k(x, y) = exp(-sum_i gamma_i * (x_i - y_i)^2).
        Parameters
        ----------
        x : torch.Tensor of shape (n, d)
        y : torch.Tensor of shape (m, d)
        Returns
        -------
        torch.Tensor of shape (n, m)
        """
        # Ensure inputs are 2D
        x = x.unsqueeze(1)          # (n, 1, d)
        y = y.unsqueeze(0)          # (1, m, d)
        diff = x - y                # (n, m, d)
        if self.gamma.numel() == diff.size(-1):
            # anisotropic gamma
            norm = torch.sum(self.gamma * diff * diff, dim=-1)
        else:
            # isotropic gamma
            norm = torch.sum(diff * diff, dim=-1) * self.gamma.item()
        return torch.exp(-norm)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: Union[float, Sequence[float]] = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of vectors.
    Parameters
    ----------
    a : Sequence[torch.Tensor], each of shape (d,)
    b : Sequence[torch.Tensor], each of shape (d,)
    gamma : scalar or sequence of scalars
    Returns
    -------
    np.ndarray of shape (len(a), len(b))
    """
    x = torch.stack(list(a))
    y = torch.stack(list(b))
    kernel = QuantumKernelMethod(gamma)
    return kernel(x, y).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
