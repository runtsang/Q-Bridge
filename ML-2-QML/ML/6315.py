"""Hybrid classical radial basis function kernel with caching and batch support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Optional

class QuantumKernelMethod__gen125(nn.Module):
    """Classical RBF kernel with optional caching and mini‑batch computation."""
    
    def __init__(self, gamma: float = 1.0, device: str | None = None, use_cache: bool = True):
        """
        Parameters
        ----------
        gamma : float, default 1.0
            Kernel width.
        device : str or torch.device, optional
            Device on which tensors are allocated. Defaults to ``'cpu'``.
        use_cache : bool, default True
            Whether to cache pairwise distances for repeated evaluations.
        """
        super().__init__()
        self.gamma = float(gamma)
        self.device = torch.device(device) if device else torch.device('cpu')
        self.use_cache = use_cache
        self._cache: dict[tuple[int, int], torch.Tensor] = {}
    
    def _pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance between two 1‑D tensors.
        """
        diff = x - y
        return torch.sum(diff * diff, dim=-1, keepdim=True)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the kernel for two 1‑D tensors.
        
        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.
            
        Returns
        -------
        torch.Tensor
            Kernel value (scalar).
        """
        x = x.to(self.device).view(-1)
        y = y.to(self.device).view(-1)
        key = (id(x), id(y))
        if self.use_cache and key in self._cache:
            return self._cache[key]
        dist = self._pairwise_distance(x, y)
        k = torch.exp(-self.gamma * dist)
        if self.use_cache:
            self._cache[key] = k
        return k.squeeze()
    
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.
        
        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.
        batch_size : int, optional
            If provided, compute the matrix in mini‑batches to reduce peak memory.
            
        Returns
        -------
        numpy.ndarray
            2‑D array of shape (len(a), len(b)).
        """
        a = [x.to(self.device).view(-1) for x in a]
        b = [y.to(self.device).view(-1) for y in b]
        n_a, n_b = len(a), len(b)
        if batch_size is None or batch_size >= n_a:
            # vectorized computation
            A = torch.stack(a)  # shape (n_a, d)
            B = torch.stack(b)  # shape (n_b, d)
            dists = torch.cdist(A, B, p=2) ** 2
            K = torch.exp(-self.gamma * dists)
            return K.cpu().numpy()
        else:
            # mini‑batch computation
            K = torch.empty((n_a, n_b), device=self.device)
            for i in range(0, n_a, batch_size):
                end = min(i + batch_size, n_a)
                A_batch = torch.stack(a[i:end])  # shape (batch, d)
                dists = torch.cdist(A_batch, torch.stack(b), p=2) ** 2
                K[i:end] = torch.exp(-self.gamma * dists)
            return K.cpu().numpy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, device={self.device}, use_cache={self.use_cache})"

__all__ = ["QuantumKernelMethod__gen125"]
