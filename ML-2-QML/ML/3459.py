"""Hybrid classical kernel combining RBF and a learnable fully‑connected mapping."""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence


class HybridKernel(nn.Module):
    """
    Hybrid kernel that blends a radial‑basis function with a learnable
    fully‑connected feature mapping. The kernel value between two
    vectors ``x`` and ``y`` is:
        k(x, y) = exp(-γ‖x−y‖²) + ⟨f(x), f(y)⟩
    where ``f`` is a linear layer followed by a hyperbolic tangent.
    ``γ`` controls the width of the RBF part; the linear layer is trained
    jointly with downstream models.
    """
    def __init__(self, gamma: float = 1.0, n_features: int = 1) -> None:
        super().__init__()
        self.gamma = gamma
        self.fc = nn.Linear(n_features, 1, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure column vectors
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        # RBF part
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=0, keepdim=True))
        # Fully‑connected mapping
        fx = self.activation(self.fc(x))
        fy = self.activation(self.fc(y))
        fc_dot = torch.mm(fx.t(), fy).squeeze()
        return rbf + fc_dot


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix for two collections of vectors.
    Parameters
    ----------
    a, b : sequences of 1‑D tensors
        Input data sets.
    gamma : float
        RBF width parameter.
    Returns
    -------
    np.ndarray
        (len(a), len(b)) kernel matrix.
    """
    kernel = HybridKernel(gamma, n_features=a[0].shape[0])
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernel", "kernel_matrix"]
