"""Hybrid classical kernel module with batched and multi‑output support.

This module extends the original RBF kernel implementation by adding:
* batch‑wise kernel matrix computation for large datasets,
* optional multi‑output (multiple gamma values) support,
* a convenience helper to train a linear SVM on the Gram matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Union
from sklearn.svm import LinearSVC

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz supporting single or multiple gamma values."""
    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__()
        if isinstance(gamma, (list, tuple, np.ndarray)):
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
        else:
            self.gamma = torch.tensor([gamma], dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two batches of vectors.
        Returns a tensor of shape (len(x), len(y), n_gamma).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (Nx, Ny, D)
        dist2 = torch.sum(diff * diff, dim=-1, keepdim=True)  # (Nx, Ny, 1)
        gamma = self.gamma.view(1, 1, -1)               # (1,1,G)
        return torch.exp(-gamma * dist2)                # (Nx, Ny, G)

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that provides batched kernel matrix computation."""
    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix between two datasets with optional batching."""
        batch_size = 1024
        x_chunks = torch.split(x, batch_size)
        y_chunks = torch.split(y, batch_size)
        mats = []
        for xc in x_chunks:
            rows = []
            for yc in y_chunks:
                rows.append(self.ansatz(xc, yc))
            rows = torch.cat(rows, dim=1)  # (len(xc), len(y), G)
            mats.append(rows)
        return torch.cat(mats, dim=0)  # (len(x), len(y), G)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: Union[float, Sequence[float]] = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two lists of tensors.
    Returns a numpy array of shape (len(a), len(b), n_gamma).
    """
    kernel = Kernel(gamma)
    a_t = torch.stack(a)
    b_t = torch.stack(b)
    mat = kernel(a_t, b_t).cpu().numpy()
    return mat

def train_svm(gram: np.ndarray, y: np.ndarray, **svm_kwargs) -> LinearSVC:
    """Train a linear SVM directly on the provided Gram matrix.
    If ``gram`` has shape (N, N, G), it is flattened to (N, N*G).
    """
    if gram.ndim == 3:
        gram = gram.reshape(gram.shape[0], -1)
    clf = LinearSVC(**svm_kwargs)
    clf.fit(gram, y)
    return clf

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "train_svm"]
