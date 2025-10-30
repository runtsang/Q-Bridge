"""Classical kernel methods with extensible kernel family."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(nn.Module):
    """A flexible classical kernel wrapper that supports multiple kernel families.

    The class can be used as a drop‑in replacement for the original
    :class:`Kernel` class while offering additional kernel types,
    automatic GPU support, and a convenient ``kernel_matrix`` helper.
    """

    def __init__(self, gamma: float = 1.0, kernel_type: str = "rbf", device: str | torch.device = "cpu") -> None:
        """
        Parameters
        ----------
        gamma : float, optional
            RBF kernel coefficient. Ignored for kernels that do not use it.
        kernel_type : str, optional
            Kernel family to use. Supported values are ``"rbf"``, ``"poly"``, and ``"linear"``.
        device : str or torch.device, optional
            Device on which to perform computations.
        """
        super().__init__()
        self.gamma = gamma
        self.kernel_type = kernel_type.lower()
        self.device = torch.device(device)

        if self.kernel_type not in {"rbf", "poly", "linear"}:
            raise ValueError(f"Unsupported kernel_type: {kernel_type!r}")

        # Hyper‑parameters for the polynomial kernel
        self.poly_degree = 3
        self.poly_coeff = 1.0

    # ------------------------------------------------------------------ #
    # Core kernel evaluation
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value ``k(x, y)``.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape ``(d,)``. The tensors are automatically
            moved to ``self.device``.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        if self.kernel_type == "rbf":
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff))
        elif self.kernel_type == "linear":
            return torch.dot(x, y)
        elif self.kernel_type == "poly":
            return (torch.dot(x, y) + self.poly_coeff) ** self.poly_degree
        else:  # pragma: no cover
            raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two batches of samples.

        Parameters
        ----------
        a, b : torch.Tensor
            Batches of shape ``(n, d)`` and ``(m, d)`` respectively.
        """
        a = a.to(self.device)
        b = b.to(self.device)

        # Broadcast and compute pairwise differences
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # shape (n, m, d)
        if self.kernel_type == "rbf":
            sq_norm = torch.sum(diff * diff, dim=-1)
            return torch.exp(-self.gamma * sq_norm)
        elif self.kernel_type == "linear":
            return torch.matmul(a, b.T)
        elif self.kernel_type == "poly":
            return (torch.matmul(a, b.T) + self.poly_coeff) ** self.poly_degree
        else:  # pragma: no cover
            raise RuntimeError("Unreachable")

    def normalize(self, gram: torch.Tensor) -> torch.Tensor:
        """
        Return a unit‑norm kernel matrix using diagonal normalization.

        Parameters
        ----------
        gram : torch.Tensor
            Kernel matrix of shape ``(n, n)``.
        """
        diag = gram.diagonal().sqrt().unsqueeze(0)
        return gram / (diag.T @ diag)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Utility to convert a torch tensor to a numpy array."""
        return tensor.detach().cpu().numpy()
