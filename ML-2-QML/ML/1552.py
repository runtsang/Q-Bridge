"""Enhanced classical RBF kernel with adaptive gamma and batch support."""

from __future__ import annotations

from typing import Sequence, Optional, Iterable

import numpy as np
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """
    RBF kernel operator supporting batched inputs.
    Computes: k(x, y) = exp(-gamma * ||x - y||^2)
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise kernel matrix between two batches.
        x: [batch_x, features]
        y: [batch_y, features]
        Returns: [batch_x, batch_y]
        """
        x = x.float()
        y = y.float()
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape: [batch_x, batch_y, features]
        dist_sq = (diff ** 2).sum(-1)
        return torch.exp(-self.gamma * dist_sq)

class Kernel(nn.Module):
    """
    Wrapper around KernalAnsatz to expose a simple API.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Efficiently compute the Gram matrix between two sequences of tensors.
    """
    if not a or not b:
        return np.array([[]], dtype=float)
    a_tensor = torch.stack([t.float() for t in a])  # shape: [len(a), features]
    b_tensor = torch.stack([t.float() for t in b])  # shape: [len(b), features]
    return Kernel(gamma)(a_tensor, b_tensor).detach().cpu().numpy()

def optimal_gamma_cv(a: Sequence[torch.Tensor], b: Optional[Sequence[torch.Tensor]] = None,
                     gamma_candidates: Iterable[float] = [0.01, 0.1, 1.0, 10.0], k: int = 5) -> float:
    """
    Rough cross‑validation to pick the bandwidth gamma.
    For each candidate gamma, compute the average kernel value on the training set.
    Returns the gamma with the largest mean kernel value.
    """
    if b is None:
        b = a
    best_gamma = gamma_candidates[0]
    best_score = -np.inf
    for gamma in gamma_candidates:
        mat = kernel_matrix(a, b, gamma)
        score = mat.mean()
        if score > best_score:
            best_score = score
            best_gamma = gamma
    return float(best_gamma)

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "optimal_gamma_cv"]
