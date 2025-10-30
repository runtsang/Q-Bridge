"""Classical radial basis function kernel utilities with learnable bandwidth."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Classical RBF kernel with a learnable `gamma` parameter.

    Parameters
    ----------
    gamma : float, optional
        Initial bandwidth of the RBF kernel.  Registered as a
        ``torch.nn.Parameter`` so it can be optimised jointly with
        downstream models.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two batches of vectors.

        The method accepts tensors of shape ``(n, d)`` and ``(m, d)``
        and returns a matrix of shape ``(n, m)``.
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)   # (n, m, d)
        sq_norm = (diff ** 2).sum(-1)            # (n, m)
        return torch.exp(-self.gamma * sq_norm)


class Kernel(nn.Module):
    """Convenience wrapper that keeps the original two‑argument API.

    The wrapper simply forwards to :class:`KernalAnsatz` and removes the
    trailing dimension added by the original seed implementation.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze(-1)


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """
    Return the Gram matrix between two lists of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors that can be stacked into shape ``(n, d)``.
    gamma : float, optional
        Initial bandwidth for the RBF kernel.

    Returns
    -------
    numpy.ndarray
        The kernel matrix of shape ``(len(a), len(b))``.
    """
    a = [torch.as_tensor(x, dtype=torch.float32) for x in a]
    b = [torch.as_tensor(y, dtype=torch.float32) for y in b]
    mat = Kernel(gamma)(torch.stack(a), torch.stack(b))
    return mat.detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
