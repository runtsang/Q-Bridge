import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class TunableKernalAnsatz(nn.Module):
    """Classical RBF kernel with a learnable gamma."""
    def __init__(self, init_gamma: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff**2, dim=-1, keepdim=True)) * self.scale

class Kernel(nn.Module):
    """Wrapper around TunableKernalAnsatz that exposes a kernel matrix."""
    def __init__(self, init_gamma: float = 1.0, scale: float = 0.5):
        super().__init__()
        self.ansatz = TunableKernalAnsatz(init_gamma, scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute Gram matrix between two sets of samples."""
        return torch.stack([self.forward(a_i, b) for a_i in a])

class NeuralHead(nn.Module):
    """Tiny feed‑forward head that maps kernel statistics to a scalar."""
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs expected to be (batch, 1)
        return self.net(inputs)

class QuantumKernelHybrid(nn.Module):
    """
    Classical‑quantum hybrid kernel model.
    Combines a learnable RBF kernel with a quantum‑encoded kernel and a tiny head.
    """
    def __init__(self, gamma: float = 1.0, hidden_dim: int = 8):
        super().__init__()
        self.kernel = Kernel(init_gamma=gamma)
        self.head = NeuralHead(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        If ``y`` is provided, compute the kernel matrix K(x, y);
        otherwise compute K(x, x) and reduce to a mean statistic.
        The result is passed through the head to obtain a scalar output.
        """
        if y is None:
            K = self.kernel(x, x).mean(dim=-1, keepdim=True)
        else:
            K = self.kernel(x, y)
        return self.head(K)

__all__ = ["TunableKernalAnsatz", "Kernel", "NeuralHead", "QuantumKernelHybrid"]
