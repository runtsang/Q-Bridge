"""Enhanced classical RBF kernel with learnable gamma and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(nn.Module):
    """Learnable RBF kernel.

    Parameters
    ----------
    gamma : float or torch.Tensor, optional
        Initial kernel width.  If a tensor is provided it becomes a
        trainable parameter; otherwise a scalar float is converted to a
        one‑element ``nn.Parameter``.
    """
    def __init__(self, gamma: float | torch.Tensor = 1.0) -> None:
        super().__init__()
        if isinstance(gamma, torch.Tensor):
            self.gamma = nn.Parameter(gamma)
        else:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel for two batch tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape ``(batch, features)``.  The function accepts
            broadcasting: if one of the inputs is a 1‑D vector it is
            treated as a singleton batch.

        Returns
        -------
        torch.Tensor
            Kernel values of shape ``(batch_x, batch_y)``.  The result is
            differentiable w.r.t. ``gamma`` and the inputs.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        diff = x[:, None, :] - y[None, :, :]
        dist2 = (diff ** 2).sum(dim=-1)
        return torch.exp(-self.gamma * dist2)


class Kernel(nn.Module):
    """Wrapper that exposes a single callable for kernel evaluation.

    It simply forwards to :class:`KernalAnsatz` but keeps the public API
    compatible with the original seed.
    """
    def __init__(self, gamma: float | torch.Tensor = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float | torch.Tensor = 1.0) -> np.ndarray:
    """Compute a Gram matrix for two collections of feature vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.  They are stacked into 2‑D tensors
        internally for efficient batch computation.
    gamma : float or torch.Tensor, optional
        Initial kernel width used by :class:`Kernel`.

    Returns
    -------
    np.ndarray
        A 2‑D NumPy array of shape ``(len(a), len(b))`` containing the
        kernel values.  The function is fully differentiable if the
        inputs are ``torch.Tensor`` with ``requires_grad=True``.
    """
    a_stack = torch.stack(a) if isinstance(a, (list, tuple)) else a
    b_stack = torch.stack(b) if isinstance(b, (list, tuple)) else b
    kernel = Kernel(gamma)
    return kernel(a_stack, b_stack).detach().cpu().numpy()
