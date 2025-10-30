"""Classical radial basis function kernel utilities with learnable bandwidth."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """RBF kernel with a learnable gamma parameter.

    The kernel is defined as
        k(x, y) = exp(-γ ||x - y||²)
    where γ is a trainable scalar. This allows the model to adapt the
    kernel width during training.
    """

    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        if gamma is None:
            gamma_val = 1.0
        else:
            gamma_val = float(gamma)
        self.gamma = nn.Parameter(torch.tensor(gamma_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix for two batches of samples.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n, d) where n is the number of samples.
        y : torch.Tensor
            Tensor of shape (m, d) where m is the number of samples.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n, m) containing k(x_i, y_j).
        """
        # Expand dimensions to enable broadcasting
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        dist_sq = (diff ** 2).sum(-1)          # (n, m)
        return torch.exp(-self.gamma * dist_sq)


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that exposes a simple interface."""

    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | None = None) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors each of shape (d,). They are first stacked
        into batches of shape (n, d) and (m, d) respectively.
    gamma : float | None
        Optional fixed gamma value. If None, the kernel uses a
        learnable parameter that defaults to 1.0.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = Kernel(gamma)
    a_stack = torch.stack([x.squeeze() for x in a])
    b_stack = torch.stack([x.squeeze() for x in b])
    return kernel(a_stack, b_stack).detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
