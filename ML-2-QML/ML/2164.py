import numpy as np
import torch
from torch import nn
from typing import Sequence

class HybridKernel(nn.Module):
    """Hybrid kernel combining RBF, polynomial, linear and optionally quantum contributions."""

    def __init__(
        self,
        gamma: float = 1.0,
        poly_degree: int = 3,
        weight: Sequence[float] | None = None,
        quantum: nn.Module | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.poly_degree = poly_degree
        self.quantum = quantum
        if weight is None:
            weight = [1.0, 1.0, 1.0, 0.0]  # RBF, poly, linear, quantum
        self.weight = torch.tensor(weight, dtype=torch.float32)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1))

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot = torch.matmul(x, y.t())
        return (dot + 1.0) ** self.poly_degree

    def _linear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.t())

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.quantum is None:
            return torch.zeros_like(self._rbf(x, y))
        return self.quantum(x, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_rbf = self._rbf(x, y)
        k_poly = self._poly(x, y)
        k_lin = self._linear(x, y)
        k_q = self._quantum_kernel(x, y)
        return (
            self.weight[0] * k_rbf
            + self.weight[1] * k_poly
            + self.weight[2] * k_lin
            + self.weight[3] * k_q
        )

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        x = torch.stack(a)
        y = torch.stack(b)
        return self.forward(x, y).detach().cpu().numpy()

__all__ = ["HybridKernel"]
