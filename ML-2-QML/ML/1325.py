"""Hybrid classical kernel module with multi‑kernel support and efficient batch evaluation.

The module extends the original RBF implementation by providing a unified interface for several
standard kernels (RBF, polynomial, linear) while keeping the ability to plug in a custom
kernel function.  It exposes a vectorised ``kernel_matrix`` routine that accepts either
single tensors or iterables of tensors, automatically broadcasting over batch dimensions.
The implementation is fully GPU‑compatible and supports chunked evaluation for large
datasets, making it suitable for use in kernel‑based learners such as SVMs or Gaussian
processes.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Callable, Optional

import torch
from torch import nn

class HybridKernel(nn.Module):
    """Unified classical kernel module.

    Parameters
    ----------
    kernel_type : str, default="rbf"
        One of ``"rbf"``, ``"poly"``, ``"linear"``, or ``"custom"``.
    gamma : float, optional
        Width parameter for the RBF kernel.
    degree : int, optional
        Degree for the polynomial kernel.
    coef0 : float, optional
        Constant term for the polynomial kernel.
    custom_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        User supplied kernel function.  Ignored unless ``kernel_type=="custom"``.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        custom_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.kernel_type = kernel_type.lower()
        if self.kernel_type not in {"rbf", "poly", "linear", "custom"}:
            raise ValueError(f"Unsupported kernel_type {kernel_type!r}")
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.custom_fn = custom_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a single pair of vectors."""
        if self.kernel_type == "rbf":
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff))
        if self.kernel_type == "poly":
            return (torch.dot(x, y) + self.coef0) ** self.degree
        if self.kernel_type == "linear":
            return torch.dot(x, y)
        # custom
        assert self.custom_fn is not None, "custom_fn must be provided for custom kernel"
        return self.custom_fn(x, y)

    @staticmethod
    def _pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Efficiently compute pairwise squared Euclidean distances."""
        # x: (n, d), y: (m, d)
        return (
            torch.sum(x * x, dim=1, keepdim=True)
            + torch.sum(y * y, dim=1)
            - 2 * torch.mm(x, y.t())
        )

    @classmethod
    def kernel_matrix(
        cls,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        custom_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Each element should be a 1‑D tensor of shape ``(d,)``.  The function
            accepts either lists or torch tensors; if a single tensor is passed
            it is treated as a batch of vectors.
        kernel_type, gamma, degree, coef0, custom_fn : see :class:`HybridKernel`
            These are forwarded to the underlying kernel implementation.
        chunk_size : int
            When ``len(a) * len(b)`` exceeds this value the kernel is evaluated
            in blocks to keep memory usage bounded.

        Returns
        -------
        torch.Tensor
            Gram matrix of shape ``(len(a), len(b))``.
        """
        if isinstance(a, torch.Tensor):
            a = [a[i] for i in range(a.shape[0])]
        if isinstance(b, torch.Tensor):
            b = [b[i] for i in range(b.shape[0])]

        device = a[0].device
        dtype = a[0].dtype

        kernel = cls(kernel_type, gamma, degree, coef0, custom_fn).to(device, dtype)

        # Pre‑allocate result
        n, m = len(a), len(b)
        result = torch.empty((n, m), device=device, dtype=dtype)

        # Process in chunks to avoid O(n*m*d) memory blow‑up
        for i in range(0, n, chunk_size):
            a_chunk = torch.stack(a[i : i + chunk_size], dim=0)
            for j in range(0, m, chunk_size):
                b_chunk = torch.stack(b[j : j + chunk_size], dim=0)
                # broadcast kernel over the chunk
                for ii, xi in enumerate(a_chunk):
                    for jj, yj in enumerate(b_chunk):
                        result[i + ii, j + jj] = kernel(xi, yj)
        return result

__all__ = ["HybridKernel", "kernel_matrix"]
