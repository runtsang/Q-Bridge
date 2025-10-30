"""Enhanced classical RBF kernel with optional learnable bandwidth and normalisation.

The class `QuantumKernelMethod` replaces the legacy `Kernel` module.
It supports:
* Fixed or learnable RBF bandwidth (`gamma`).
* Optional mean‑zero, unit‑variance normalisation via `nn.BatchNorm1d`.
* Vectorised `kernel_matrix` routine that accepts lists of tensors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with optional learnable gamma and normalisation.

    Parameters
    ----------
    gamma : float, optional
        Initial RBF bandwidth.  If ``learnable=True`` this value is wrapped
        in a `nn.Parameter` and optimised during training.
    learnable : bool, default=False
        Whether the bandwidth should be a trainable parameter.
    normalize : bool, default=True
        If ``True`` inputs are normalised to zero mean and unit variance
        before kernel evaluation.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        learnable: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        self.normalize = normalize
        if self.learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        if self.normalize:
            # Simple per‑feature batch normalisation; works on 2‑D tensors.
            self.norm = nn.BatchNorm1d(1)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            # Reshape to (batch, 1, features) for BatchNorm1d
            x = x.unsqueeze(1)
            x = self.norm(x)
            return x.squeeze(1)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.

        Returns
        -------
        torch.Tensor
            Kernel value as a scalar tensor.
        """
        x = self._preprocess(x)
        y = self._preprocess(y)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        np.ndarray
            2‑D array of shape (len(a), len(b)).
        """
        mat = torch.empty((len(a), len(b)), dtype=torch.float32)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat.cpu().numpy()


def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """
    Compatibility shim that mirrors the legacy function.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
    gamma : float, optional
        Bandwidth for the fixed RBF kernel.

    Returns
    -------
    np.ndarray
    """
    kernel = QuantumKernelMethod(gamma=gamma, learnable=False, normalize=False)
    return kernel.kernel_matrix(a, b)


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
