"""Enhanced classical RBF kernel with learnable gamma and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Learnable RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Initial value of the RBF width. If ``learnable`` is True, this
        value becomes a trainable parameter.
    learnable : bool, default=False
        Whether ``gamma`` should be optimized during training.
    """

    def __init__(self, gamma: float = 1.0, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two batches.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape ``(batch, features)``. The function is
            broadcastable to any shape that is compatible with the
            Euclidean distance computation.

        Returns
        -------
        torch.Tensor
            Kernel values of shape ``(batch_x, batch_y)``.
        """
        x = x.float()
        y = y.float()
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (batch_x, batch_y, features)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (batch_x, batch_y)
        return torch.exp(-self.gamma * dist_sq)

class Kernel(nn.Module):
    """Convenience wrapper that exposes the same API as the quantum kernel."""

    def __init__(self, gamma: float = 1.0, learnable: bool = False) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, learnable)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of samples.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Each element should be a 1‑D tensor representing a sample.
    gamma : float, optional
        RBF width used if a non‑learnable kernel is required.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape ``(len(a), len(b))``.
    """
    X = torch.stack(list(a))
    Y = torch.stack(list(b))
    kernel = Kernel(gamma)
    K = kernel(X, Y)
    return K.detach().cpu().numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
