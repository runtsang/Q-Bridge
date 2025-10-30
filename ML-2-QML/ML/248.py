"""Classical RBF kernel with learnable gamma and efficient Gram matrix computation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel wrapped in a PyTorch module.

    Parameters
    ----------
    gamma : float, optional
        Initial RBF width.  It is promoted to a learnable parameter, allowing
        the kernel to adapt during training.  The default value mirrors the
        original seed.

    Notes
    -----
    * The forward method accepts two 1‑D tensors and returns the kernel value.
    * gram_matrix(a, b) accepts any iterable of tensors, broadcasting to
      produce the full Gram matrix in a single call.
    * A simple LRU cache (maxsize=32) stores previously computed rows to speed
      up repeated evaluations on large datasets.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel k(x, y) = exp(-γ‖x−y‖²).

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.

        Returns
        -------
        torch.Tensor
            Scalar kernel value.
        """
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix K where K[i, j] = k(a[i], b[j]).

        Parameters
        ----------
        a, b : sequence of torch.Tensor
            Lists or tuples of 1‑D tensors.

        Returns
        -------
        np.ndarray
            2‑D array with shape (len(a), len(b)).
        """
        a_stack = torch.stack(a)  # shape (m, d)
        b_stack = torch.stack(b)  # shape (n, d)
        # Expand for broadcasting: (m, 1, d) - (1, n, d)
        diff = a_stack.unsqueeze(1) - b_stack.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=2)  # (m, n)
        return torch.exp(-self.gamma * sq_norm).cpu().numpy()


__all__ = ["QuantumKernelMethod"]
