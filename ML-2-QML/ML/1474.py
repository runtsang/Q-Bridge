"""Classical radial basis function kernel utilities with feature learning and learnable gamma."""
from __future__ import annotations

from typing import Sequence
import math
import numpy as np
import torch
from torch import nn

class FeatureExtractor(nn.Module):
    """Learnable dense layer to transform raw inputs before RBF."""
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class KernalAnsatz(nn.Module):
    """RBF kernel with learnable width."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist_sq)

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz` and an optional feature extractor."""
    def __init__(self, gamma: float = 1.0, in_dim: int | None = None, hidden_dim: int = 32):
        super().__init__()
        self.extractor = FeatureExtractor(in_dim, hidden_dim) if in_dim is not None else nn.Identity()
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        y = self.extractor(y)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0,
                  in_dim: int | None = None, hidden_dim: int = 32) -> np.ndarray:
    """Compute the Gram matrix between datasets ``a`` and ``b`` using the extended kernel."""
    x = torch.stack(a)
    y = torch.stack(b)
    extractor = FeatureExtractor(in_dim, hidden_dim) if in_dim is not None else nn.Identity()
    x = extractor(x)
    y = extractor(y)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist_sq = torch.sum(diff * diff, dim=-1)
    return np.exp(-gamma * dist_sq).numpy()

__all__ = ["FeatureExtractor", "KernalAnsatz", "Kernel", "kernel_matrix"]
