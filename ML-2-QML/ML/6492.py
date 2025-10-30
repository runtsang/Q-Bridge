"""Enhanced classical kernel method with trainable parameters and MLP‑based kernel learning."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class KernalAnsatz(nn.Module):
    """Base kernel ansatz supporting RBF, linear, polynomial, and MLP kernels.

    The class is fully differentiable and can be trained with standard PyTorch optimizers.
    """
    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coeff0: float = 1.0,
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_type = kernel_type.lower()
        if self.kernel_type == "rbf":
            self.gamma = nn.Parameter(torch.tensor(gamma))
        elif self.kernel_type == "linear":
            pass
        elif self.kernel_type == "poly":
            self.degree = degree
            self.coeff0 = coeff0
        elif self.kernel_type == "mlp":
            self.hidden_sizes = hidden_sizes or (32, 16)
            self._mlp_built = False
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure 2‑D tensors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        if self.kernel_type == "rbf":
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        elif self.kernel_type == "linear":
            return torch.sum(x * y, dim=-1, keepdim=True)
        elif self.kernel_type == "poly":
            return (torch.sum(x * y, dim=-1, keepdim=True) + self.coeff0) ** self.degree
        else:  # mlp
            if not self._mlp_built:
                input_dim = x.shape[-1] * 2
                self.mlp = self._build_mlp(input_dim)
                self._mlp_built = True
            concat = torch.cat([x, y], dim=-1)
            return self.mlp(concat).view(-1, 1)


class Kernel(nn.Module):
    """Convenience wrapper that exposes the same API as the original seed."""
    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coeff0: float = 1.0,
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(
            kernel_type=kernel_type,
            gamma=gamma,
            degree=degree,
            coeff0=coeff0,
            hidden_sizes=hidden_sizes,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coeff0: float = 1.0,
    hidden_sizes: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = Kernel(
        kernel_type=kernel_type,
        gamma=gamma,
        degree=degree,
        coeff0=coeff0,
        hidden_sizes=hidden_sizes,
    )
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
