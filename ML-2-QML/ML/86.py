"""Extended radial basis function kernel with neural‑network weight sharing."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """A trainable RBF kernel that uses a shared weight matrix for multi‑output regression."""
    def __init__(self, gamma: float = 1.0, out_dim: int = 1, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.gamma = gamma
        self.out_dim = out_dim
        # weight shape: (1, 1) to uniformly scale all dimensions; can be extended for per‑output scaling
        self.weight = nn.Parameter(torch.randn(1, 1))
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two batches of vectors."""
        x = x.to(self.device)
        y = y.to(self.device)
        diff = (x - y) * self.weight
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def train_kernel(self, x: torch.Tensor, y: torch.Tensor,
                     targets: torch.Tensor,
                     lr: float = 1e-3, epochs: int = 200) -> None:
        """End‑to‑end optimisation of the weight matrix."""
        self.train()
        optimizer = torch.optim.Adam([self.weight], lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(x, y).squeeze()
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inference wrapper that detaches the result."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  out_dim: int = 1) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    model = QuantumKernelMethod(gamma=gamma, out_dim=out_dim)
    return np.array([[model.forward(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
