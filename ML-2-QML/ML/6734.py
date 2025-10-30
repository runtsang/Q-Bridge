"""Hybrid classical kernel with trainable neural feature map and RBF kernel."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Neural feature map: a small MLP that maps input features into a higher‑dimensional space."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int | None = None) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        return self.net(x)

class Kernel(nn.Module):
    """Hybrid kernel that applies a neural feature map then an RBF kernel."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int | None = None, gamma: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(input_dim, hidden_dim, out_dim)
        if gamma is None:
            self.gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y shape: (batch, input_dim) or (input_dim,)
        x = x if x.ndim == 2 else x.unsqueeze(0)
        y = y if y.ndim == 2 else y.unsqueeze(0)
        fx = self.ansatz(x)  # (batch_x, out_dim)
        fy = self.ansatz(y)  # (batch_y, out_dim)
        # compute pairwise squared Euclidean distances using broadcasting
        diff = fx.unsqueeze(1) - fy.unsqueeze(0)  # (batch_x, batch_y, out_dim)
        sq_dist = torch.sum(diff ** 2, dim=-1)     # (batch_x, batch_y)
        return torch.exp(-self.gamma * sq_dist)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], input_dim: int, hidden_dim: int = 64, out_dim: int | None = None, gamma: float | None = None) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors using the hybrid kernel."""
    kernel = Kernel(input_dim, hidden_dim, out_dim, gamma)
    # Convert list of tensors to a single tensor
    A = torch.stack(a).float()
    B = torch.stack(b).float()
    K = kernel(A, B).cpu().numpy()
    return K

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
