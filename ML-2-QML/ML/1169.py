import numpy as np
import torch
from torch import nn
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical RBF kernel with batched support and optional normalization.

    Parameters
    ----------
    gamma : float, optional
        Kernel width parameter. Default 1.0.
    normalize : bool, optional
        If True, normalizes the kernel matrix to have unit diagonal.
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape (n_samples_x, n_features)
        y : torch.Tensor
            Shape (n_samples_y, n_features)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        # Ensure 2D tensors
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        # Compute squared Euclidean distances
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n_x, n_y, d)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (n_x, n_y)
        k = torch.exp(-self.gamma * dist_sq)
        if self.normalize:
            # Normalize to unit diagonal
            diag = torch.diag(k)
            norm_factor = torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0))
            k = k / norm_factor
        return k

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper around :meth:`forward`."""
        return self.forward(a, b)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        Sequence of vectors (each shape (n_features,))
    b : Sequence[torch.Tensor]
        Sequence of vectors.
    gamma : float, optional
        Kernel width. Default 1.0.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b))
    """
    kernel = RBFKernel(gamma)
    return np.array([[kernel(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["RBFKernel", "kernel_matrix"]
