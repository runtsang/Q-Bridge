"""Enhanced classical RBF kernel implementation with learnable gamma and GPU support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """
    Classical radial basis function (RBF) kernel with optional learnable gamma.

    Parameters
    ----------
    gamma : float or torch.nn.Parameter, default=1.0
        Width of the kernel. If a torch.nn.Parameter is provided it will be
        updated during training.
    device : str or torch.device, default='cpu'
        Device on which tensors are stored.

    The forward method returns a pairwise kernel matrix between two input
    tensors of shape (n_samples, n_features).
    """

    def __init__(self, gamma: float | nn.Parameter = 1.0, device: str | torch.device = "cpu") -> None:
        super().__init__()
        if isinstance(gamma, float):
            self.gamma = nn.Parameter(torch.tensor(gamma, device=device, dtype=torch.float))
        else:
            self.gamma = gamma
        self.device = device

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (nX, nY, d)
        return torch.exp(-self.gamma * (diff * diff).sum(-1))

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
        """
        Compute the kernel Gram matrix and return it as a NumPy array.

        Parameters
        ----------
        X : torch.Tensor
        Y : torch.Tensor

        Returns
        -------
        numpy.ndarray
            Pairwise kernel matrix of shape (len(X), len(Y)).
        """
        return self.forward(X, Y).cpu().numpy()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Legacy helper that mimics the original API.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
    b : Sequence[torch.Tensor]
    gamma : float, optional

    Returns
    -------
    numpy.ndarray
        Pairwise kernel matrix.
    """
    kernel = QuantumKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernel", "kernel_matrix"]
