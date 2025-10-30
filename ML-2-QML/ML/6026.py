"""Enhanced classical RBF kernel with learnable gamma and simple training routine."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss


class KernalAnsatz(nn.Module):
    """RBF kernel with a learnable gamma parameter."""
    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        if gamma is None:
            gamma = 1.0
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
    def extra_repr(self) -> str:
        return f"gamma={self.gamma.item():.4f}"


class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with optional training of gamma."""
    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        mat = np.zeros((len(a), len(b)), dtype=float)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x.unsqueeze(0), y.unsqueeze(0)).item()
        return mat
    def kernel_matrix_torch(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x.unsqueeze(0), y.unsqueeze(0)).squeeze()
        return mat
    def train_gamma(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], target: np.ndarray,
                    lr: float = 0.01, epochs: int = 200) -> None:
        """Train the gamma parameter to match a target Gram matrix."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        for epoch in range(epochs):
            optimizer.zero_grad()
            mat = self.kernel_matrix_torch(a, b)
            loss = mse_loss(mat, target_tensor)
            loss.backward()
            optimizer.step()
            if epoch % max(1, (epochs // 10)) == 0:
                print(f"Epoch {epoch}, loss={loss.item():.6f}")


__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
