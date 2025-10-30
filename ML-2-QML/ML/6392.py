"""Hybrid classical kernel module with trainable parameters and batch support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with a learnable gamma parameter.
    Provides end‑to‑end differentiable kernel evaluation for batch data.
    """

    def __init__(self, initial_gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of samples.
        x, y: shape (batch, features)
        Returns: shape (batch, batch)
        """
        # Expand to compute pairwise differences
        x_exp = x.unsqueeze(1)  # (B, 1, F)
        y_exp = y.unsqueeze(0)  # (1, B, F)
        diff = x_exp - y_exp   # (B, B, F)
        sq_norm = torch.sum(diff * diff, dim=-1)  # (B, B)
        return torch.exp(-self.gamma * sq_norm)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.
        Each element in a and b is a 1‑D tensor of features.
        """
        a_stack = torch.stack(a)  # (N, F)
        b_stack = torch.stack(b)  # (M, F)
        return self.forward(a_stack, b_stack).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
