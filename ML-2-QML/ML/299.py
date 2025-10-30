"""Enhanced classical RBF kernel with learnable bandwidth and batch support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Compatibility alias for original KernalAnsatz. Acts as a simple static RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Learnable RBF kernel with optional batch and multi‑output support."""
    def __init__(self, gamma: float | Sequence[float] = 1.0, learnable: bool = False) -> None:
        super().__init__()
        if isinstance(gamma, (list, tuple)):
            gamma_tensor = torch.tensor(gamma, dtype=torch.float32, requires_grad=learnable)
        else:
            gamma_tensor = torch.tensor([gamma], dtype=torch.float32, requires_grad=learnable)
        self.gamma = nn.Parameter(gamma_tensor, requires_grad=learnable)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between input tensors x and y.
        Supports batched inputs and returns a 3‑D tensor when multiple gammas are provided.
        """
        # Ensure inputs are 2‑D: (batch, features)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # Expand dimensions for broadcasting
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)  # shape (batch_x, batch_y, features)
        dist2 = torch.sum(diff * diff, dim=-1)  # shape (batch_x, batch_y)

        # Expand gamma for broadcasting
        gamma = self.gamma.view(1, 1, -1)  # shape (1, 1, num_gammas)
        k = torch.exp(-gamma * dist2.unsqueeze(-1))  # shape (batch_x, batch_y, num_gammas)

        # Collapse the last dimension if only one gamma
        if k.shape[-1] == 1:
            return k.squeeze(-1)
        return k

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two sequences of tensors."""
        a_tensor = torch.stack(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.stack(b) if not isinstance(b, torch.Tensor) else b
        with torch.no_grad():
            kernel = self(a_tensor, b_tensor)
        return kernel.cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel"]
