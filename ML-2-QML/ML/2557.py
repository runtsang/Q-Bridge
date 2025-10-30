from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Classical hybrid kernel that combines an RBF kernel with an optional
    fully‑connected feature mapper.  The mapper can be trained
    independently and the resulting representations are fed into the
    RBF kernel.  This follows the structure of the original
    `KernalAnsatz/Kernel` pair while adding a learnable mapping layer.
    """

    def __init__(self, gamma: float = 1.0, n_features: int | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.feature_mapper = nn.Linear(n_features, 1) if n_features else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.feature_mapper:
            x = self.feature_mapper(x)
            y = self.feature_mapper(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of samples."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernelMethod"]
