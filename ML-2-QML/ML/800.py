"""Hybrid classical kernel module with trainable gamma and mini‑batch support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz with a learnable gamma parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper for the RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    def gram_matrix(self,
                    X: torch.Tensor,
                    Y: torch.Tensor,
                    batch_size: int = 256) -> torch.Tensor:
        """Compute the Gram matrix in a memory‑friendly way."""
        n, _ = X.shape
        m, _ = Y.shape
        K = torch.empty((n, m), dtype=X.dtype, device=X.device)
        for i in range(0, n, batch_size):
            x_batch = X[i:i+batch_size]
            for j in range(0, m, batch_size):
                y_batch = Y[j:j+batch_size]
                K[i:i+batch_size, j:j+batch_size] = self(x_batch, y_batch)
        return K

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  batch_size: int = 256) -> np.ndarray:
    """Return the Gram matrix as a NumPy array."""
    kernel = Kernel(gamma)
    X = torch.stack([x.squeeze() for x in a])
    Y = torch.stack([y.squeeze() for y in b])
    return kernel.gram_matrix(X, Y).cpu().numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
