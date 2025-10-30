"""Extended classical kernel module with learnable parameters and batch support.

This implementation builds on the original RBF kernel but adds:
* Learnable hyper‑parameters (gamma, sigma) as torch Parameters.
* Batch‑wise kernel matrix construction with broadcasting.
* Automatic differentiation support for downstream learning.
* Optional GPU acceleration when CUDA is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with learnable hyper‑parameters.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the RBF exponent coefficient.
    sigma : float, optional
        Initial value for the Gaussian width.
    device : str | torch.device, optional
        Target device. Defaults to CUDA if available.
    """

    def __init__(self, gamma: float = 1.0, sigma: float = 1.0, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape (n,) or (batch, n).

        Returns
        -------
        torch.Tensor
            Kernel value(s) of shape (batch,) or scalar.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm / (2 * self.sigma ** 2)).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Gram matrix between two sequences of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(a), len(b)).
        """
        A = torch.stack([x.to(self.device) for x in a])  # shape (m, d)
        B = torch.stack([y.to(self.device) for y in b])  # shape (n, d)
        diff = A.unsqueeze(1) - B.unsqueeze(0)  # shape (m, n, d)
        sq_norm = torch.sum(diff * diff, dim=-1)  # shape (m, n)
        return torch.exp(-self.gamma * sq_norm / (2 * self.sigma ** 2))

    def to_numpy(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """
        Helper to compute the kernel matrix and return a NumPy array.

        Parameters
        ----------
        a, b : Sequence[np.ndarray]
            Sequences of 1‑D NumPy arrays.

        Returns
        -------
        np.ndarray
            Kernel matrix as a NumPy array.
        """
        torch_a = [torch.from_numpy(x).float() for x in a]
        torch_b = [torch.from_numpy(y).float() for y in b]
        return self.kernel_matrix(torch_a, torch_b).cpu().numpy()


__all__ = ["QuantumKernelMethod"]
