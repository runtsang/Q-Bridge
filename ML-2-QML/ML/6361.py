"""Hybrid classical kernel module with learnable RBF parameters and batch support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class KernalAnsatz(nn.Module):
    """Learnable RBF kernel ansatz.

    The kernel has the form
        k(x, y) = exp(-γ * ||x - y||²),
    where γ is produced by a small neural network to ensure positivity.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        """
        Args:
            input_dim: Dimensionality of the input vectors.
            hidden_dim: Hidden dimension of the gamma‑network.
        """
        super().__init__()
        # Small MLP to produce a positive gamma
        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures positivity
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise RBF kernel between all rows of x and y.

        Args:
            x: Tensor of shape (n_samples_x, d)
            y: Tensor of shape (n_samples_y, d)

        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        # Compute gamma from x (or y); use the first sample's gamma for simplicity
        gamma = self.gamma_net(x[0:1])  # shape (1,1)
        gamma = gamma.squeeze() + 1e-6  # avoid zero

        # Compute squared Euclidean distances
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (n_x, n_y, d)
        dist_sq = torch.sum(diff * diff, dim=-1)  # shape (n_x, n_y)
        return torch.exp(-gamma * dist_sq)


class Kernel(nn.Module):
    """Wrapper that provides a batched kernel interface."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        """
        Args:
            input_dim: Dimensionality of the input vectors.
            hidden_dim: Hidden dimension of the gamma‑network.
        """
        super().__init__()
        self.ansatz = KernalAnsatz(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the full kernel matrix between two batches.

        Args:
            x: Tensor of shape (n_samples_x, d)
            y: Tensor of shape (n_samples_y, d)

        Returns:
            Tensor of shape (n_samples_x, n_samples_y)
        """
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], hidden_dim: int = 16) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors.

    Args:
        a: Sequence of tensors of shape (d,)
        b: Sequence of tensors of shape (d,)
        hidden_dim: Hidden dimension of the gamma‑network.

    Returns:
        NumPy array of shape (len(a), len(b))
    """
    kernel = Kernel(input_dim=a[0].shape[0], hidden_dim=hidden_dim)
    a_batch = torch.stack(a)  # shape (len(a), d)
    b_batch = torch.stack(b)  # shape (len(b), d)
    with torch.no_grad():
        return kernel(a_batch, b_batch).cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
