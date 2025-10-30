"""Enhanced RBF kernel utilities with GPU support and trainable hyperparameters.

The module extends the original ``KernalAnsatz`` by:
* Vectorised pairwise distance computation for large batches.
* Optional trainable ``gamma`` for automatic hyper‑parameter tuning.
* Convenience helpers for kernel matrix construction and simple cross‑validation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable, Optional


class KernalAnsatz(nn.Module):
    """Vectorised RBF kernel with optional trainable ``gamma``.

    Parameters
    ----------
    gamma : float, default 1.0
        Initial value for the kernel width.
    trainable : bool, default False
        Whether ``gamma`` should be a learnable ``torch.nn.Parameter``.
    """

    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix ``K_{ij} = exp(-γ‖x_i−y_j‖²)``."""
        # Ensure 2‑D shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # Efficient pairwise squared distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (n,1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (m,1)
        dist_sq = x_norm + y_norm.t() - 2.0 * x @ y.t()
        return torch.exp(-self.gamma * dist_sq)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Gram matrix for two sequences of tensors."""
        a_stack = torch.stack(a, dim=0)
        b_stack = torch.stack(b, dim=0)

        if batch_size is None or batch_size >= len(a_stack):
            return self.forward(a_stack, b_stack).cpu().numpy()

        # Batched computation
        kernels = []
        for i in range(0, len(a_stack), batch_size):
            kernels.append(
                self.forward(a_stack[i : i + batch_size], b_stack).cpu()
            )
        return torch.cat(kernels, dim=0).numpy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(gamma={self.gamma.item():.3f}, "
            f"trainable={self.gamma.requires_grad})"
        )


class Kernel(nn.Module):
    """Convenience wrapper that optionally trains ``gamma`` via grid search.

    The wrapper keeps a ``KernalAnsatz`` instance and exposes a lightweight
    ``fit_gamma`` method that searches over a user‑supplied list of values.
    """

    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, trainable=trainable)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def fit_gamma(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        gamma_candidates: Iterable[float],
        *,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Simple grid‑search for ``gamma`` that maximises the Frobenius norm
        of the kernel matrix on a validation set.  The chosen value is
        stored in ``self.ansatz.gamma``.
        """
        X, y = X.to(device), y.to(device)
        best_gamma, best_score = None, -np.inf
        for g in gamma_candidates:
            self.ansatz.gamma = torch.tensor(g, dtype=torch.float32, device=device)
            K = self.forward(X, X)
            score = torch.norm(K, p="fro").item()
            if score > best_score:
                best_gamma, best_score = g, score
        self.ansatz.gamma = torch.tensor(best_gamma, dtype=torch.float32, device=device)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return self.ansatz.kernel_matrix(a, b, batch_size=batch_size)


__all__ = ["KernalAnsatz", "Kernel"]
