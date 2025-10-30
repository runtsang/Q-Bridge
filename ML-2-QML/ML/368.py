"""Enhanced classical radial basis function kernel utilities with batch support and hyper‑parameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn


@dataclass
class KernalAnsatz(nn.Module):
    """Vectorised RBF kernel with optional gamma optimisation.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the Gaussian kernel.  A higher value yields a more
        concentrated similarity measure.

    Notes
    -----
    The forward method accepts either two tensors of shape ``(n, d)``
    or two 1‑D tensors.  Internally the operation is fully vectorised
    using broadcasting, which is significantly faster than the
    original pairwise loop.
    """
    gamma: float = 1.0

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure 2‑D tensors
        x = x if x.ndim == 2 else x.unsqueeze(0)
        y = y if y.ndim == 2 else y.unsqueeze(0)

        # Broadcast to compute pairwise Euclidean distances
        diff = x[:, None, :] - y[None, :, :]
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)


class Kernel(nn.Module):
    """Convenience wrapper that exposes the same API as the original
    ``Kernel`` class but with a fully vectorised implementation.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the Gaussian kernel.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(
    a: Union[Sequence[torch.Tensor], torch.Tensor],
    b: Union[Sequence[torch.Tensor], torch.Tensor],
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute the Gram matrix between two datasets.

    Parameters
    ----------
    a, b : iterable of tensors or 2‑D tensors
        Input datasets.  ``a`` and ``b`` may be lists of 1‑D tensors
        or already batched tensors of shape ``(n, d)``.
    gamma : float, default=1.0
        Kernel width.

    Returns
    -------
    np.ndarray
        Gram matrix of shape ``(len(a), len(b))``.
    """
    if isinstance(a, (list, tuple)):
        a = torch.stack(a)
    if isinstance(b, (list, tuple)):
        b = torch.stack(b)

    kernel = Kernel(gamma)
    return kernel(a, b).detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
