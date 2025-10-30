"""Hybrid kernel layer combining classical RBF and a trainable linear head."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class HybridKernelLayer(nn.Module):
    """
    Classical implementation of a kernel‑based layer.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF kernel bandwidth.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.linear: Optional[nn.Linear] = None

    def _kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF Gram matrix between X and Y."""
        diff = X[:, None, :] - Y[None, :, :]
        sq_norm = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples_x, d).
        Y : torch.Tensor
            Reference tensor of shape (n_samples_y, d).

        Returns
        -------
        torch.Tensor
            Output of shape (n_samples_x,) after linear projection.
        """
        K = self._kernel_matrix(X, Y)

        if self.linear is None:
            # lazily initialise linear head once kernel dimension is known
            self.linear = nn.Linear(K.size(-1), 1, bias=True)

        return self.linear(K).squeeze(-1)

    def kernel_matrix(self, a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper that returns a NumPy array."""
        kernel = self._kernel_matrix(torch.stack(a), torch.stack(b))
        return kernel.detach().cpu().numpy()


__all__ = ["HybridKernelLayer"]
