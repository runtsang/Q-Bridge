"""Extended classical RBF kernel with trainable gamma and batch support."""

import numpy as np
import torch
from torch import nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Trainable RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the inverse width. Stored as a learnable parameter.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix K_{ij} = exp(-gamma * ||x_i - y_j||^2).

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n_samples_x, n_features).
        y : torch.Tensor
            Tensor of shape (n_samples_y, n_features).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n_samples_x, n_samples_y).
        """
        diff = x[:, None, :] - y[None, :, :]
        sq_norm = (diff ** 2).sum(dim=2)
        return torch.exp(-self.gamma * sq_norm)

class Kernel(nn.Module):
    """Wrapper that provides a convenient API for the RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the Gram matrix between `x` and `y`.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            The kernel matrix of shape (n_samples_x, n_samples_y).
        """
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the kernel matrix between two collections of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors; each tensor is expected to be 1‑D.
    gamma : float, optional
        Initial inverse width for the RBF kernel.

    Returns
    -------
    np.ndarray
        Kernel matrix as a NumPy array.
    """
    kernel = Kernel(gamma)
    a_flat = [x.reshape(-1) for x in a]
    b_flat = [y.reshape(-1) for y in b]
    return np.array([[kernel(x, y).item() for y in b_flat] for x in a_flat])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]

# Simple test harness
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(5, 3)
    Y = torch.randn(4, 3)
    k = Kernel(gamma=0.5)
    print("Kernel matrix:\\n", k(X, Y).detach().numpy())
