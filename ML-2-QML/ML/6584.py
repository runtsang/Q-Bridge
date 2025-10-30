"""Hybrid classical kernel module with learnable hyper‑parameters and experimental pipeline."""
from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import torch
from torch import nn, optim

__all__ = [
    "QuantumKernelMethod",
    "KernelFactory",
    "train_gammas",
    "kernel_matrix",
]

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with learnable gamma hyper‑parameter.

    The class retains compatibility with the original interface but now
    exposes a trainable `gamma` parameter that can be optimised via
    gradient‑based methods.  The kernel can be evaluated on batches of
    data and the Gram matrix can be computed efficiently using
    broadcasting.
    """
    def __init__(self, gamma: float = 1.0, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.learnable = learnable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two batches of vectors."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        diff = x[:, None, :] - y[None, :, :]
        sqdist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sqdist)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Evaluate the Gram matrix between two lists of tensors."""
    kernel = QuantumKernelMethod(gamma=gamma, learnable=False)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def train_gammas(
    data: Sequence[torch.Tensor],
    labels: Sequence[int],
    lr: float = 1e-2,
    epochs: int = 100,
    loss_fn: Any = nn.MSELoss(),
) -> float:
    """Simple gradient‑based training loop for the gamma hyper‑parameter."""
    X = torch.stack(data)
    y = torch.tensor(labels, dtype=torch.float32)
    model = QuantumKernelMethod(gamma=1.0, learnable=True)
    optimizer = optim.Adam([model.gamma], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        K = model(X, X)
        loss = loss_fn(K.diag(), y)
        loss.backward()
        optimizer.step()
    return model.gamma.item()

class KernelFactory:
    """Factory to instantiate either a classical or quantum kernel."""
    def __init__(self, use_quantum: bool = False, **kwargs: Any) -> None:
        self.use_quantum = use_quantum
        self.kwargs = kwargs

    def get_kernel(self) -> nn.Module:
        if self.use_quantum:
            # Lazy import to avoid circular dependency
            from. import QuantumKernelMethod as QuantumKernel
            return QuantumKernel(**self.kwargs)
        else:
            return QuantumKernelMethod(**self.kwargs)
