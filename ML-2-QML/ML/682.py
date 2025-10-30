"""Hybrid classical kernel module with advanced features.

The new implementation supports:
* Batch‑wise RBF kernels for large datasets.
* Optional PCA dimensionality reduction.
* Kernel‑alignment diagnostics to check similarity of kernels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from typing import Sequence

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz supporting batched inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (m, d).
        y : torch.Tensor
            Tensor of shape (n, d).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (m, n).
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (m, n, d)
        sq_dist = torch.sum(diff ** 2, dim=-1)   # (m, n)
        return torch.exp(-self.gamma * sq_dist)

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First collection of vectors.
    b : Sequence[torch.Tensor]
        Second collection of vectors.
    gamma : float, default 1.0
        RBF width parameter.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    x = torch.stack(a)
    y = torch.stack(b)
    kernel = Kernel(gamma)
    return kernel(x, y).detach().cpu().numpy()

def pca_transform(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Reduce dimensionality of X using PCA.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (samples, features).
    n_components : int
        Number of principal components.

    Returns
    -------
    np.ndarray
        Transformed data of shape (samples, n_components).
    """
    pca = PCA(n_components=n_components, svd_solver='auto')
    return pca.fit_transform(X)

def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Compute the alignment between two kernel matrices.

    Alignment is defined as
        trace(K1 @ K2) / sqrt(trace(K1 @ K1) * trace(K2 @ K2))

    Parameters
    ----------
    K1 : np.ndarray
    K2 : np.ndarray

    Returns
    -------
    float
        Kernel alignment value in [0, 1].
    """
    num = np.trace(K1 @ K2)
    denom = np.sqrt(np.trace(K1 @ K1) * np.trace(K2 @ K2))
    return float(num / denom) if denom!= 0 else 0.0

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "pca_transform", "kernel_alignment"]
