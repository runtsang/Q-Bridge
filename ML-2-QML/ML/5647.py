"""Enhanced classical RBF kernel with learnable gamma and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """
    Classical radial basis function (RBF) kernel with optional learnable gamma.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the RBF width. If ``learnable_gamma`` is True,
        gamma becomes a trainable parameter.
    learnable_gamma : bool, default False
        Whether to treat gamma as a trainable parameter.
    device : str or torch.device, default 'cpu'
        Device on which the kernel will be evaluated.
    regularization : str, optional
        One of ``None`` or ``'l2'`` to add λ * I to the Gram matrix.
    reg_weight : float, default 0.0
        Regularization weight λ.

    Notes
    -----
    The kernel can handle batched inputs: ``x`` and ``y`` may have shape
    ``(batch, n_features)`` or ``(n_samples, n_features)``. The
    resulting kernel matrix will have shape ``(batch_x, batch_y)``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        learnable_gamma: bool = False,
        device: str | torch.device = "cpu",
        regularization: str | None = None,
        reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        if learnable_gamma:
            self.gamma = nn.Parameter(
                torch.tensor(gamma, dtype=torch.float32, device=self.device)
            )
        else:
            self.register_buffer(
                "gamma", torch.tensor(gamma, dtype=torch.float32, device=self.device)
            )
        self.learnable_gamma = learnable_gamma
        self.regularization = regularization
        self.reg_weight = reg_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of samples.

        Parameters
        ----------
        x : torch.Tensor, shape (m, d) or (1, d)
        y : torch.Tensor, shape (n, d) or (1, d)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (m, n).
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (m, n, d)
        dist_sq = (diff ** 2).sum(dim=-1)  # shape (m, n)
        K = torch.exp(-self.gamma * dist_sq)
        if self.regularization == "l2":
            I = torch.eye(K.shape[0], device=self.device)
            K = K + self.reg_weight * I
        return K

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
        b : Sequence[torch.Tensor]

        Returns
        -------
        np.ndarray
            NumPy array of shape (len(a), len(b)).
        """
        m, n = len(a), len(b)
        K = torch.zeros((m, n), device=self.device)
        for i in range(m):
            for j in range(n):
                K[i, j] = self.forward(a[i], b[j]).item()
        return K.cpu().numpy()

__all__ = ["QuantumKernelMethod"]
