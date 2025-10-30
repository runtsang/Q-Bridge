"""HybridSamplerQNN: classical sampler network with optional RBF kernel.

This module defines a PyTorch‑based sampler that combines a feed‑forward
network with a radial‑basis‑function kernel for similarity estimation.
The design mirrors the original SamplerQNN while adding a kernel layer
that can be used for downstream kernel‑based learning tasks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classic RBF kernel as a PyTorch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridSamplerQNN(nn.Module):
    """Classical sampler network with optional RBF kernel support."""
    def __init__(self, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.kernel = Kernel(gamma) if self.use_kernel else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        probs = F.softmax(self.net(inputs), dim=-1)
        if self.kernel is not None:
            # return both probability distribution and kernel similarity
            return probs, self.kernel(probs, probs)
        return probs

def SamplerQNN(use_kernel: bool = False, gamma: float = 1.0) -> HybridSamplerQNN:
    """Factory returning a HybridSamplerQNN instance."""
    return HybridSamplerQNN(use_kernel=use_kernel, gamma=gamma)

__all__ = ["SamplerQNN", "HybridSamplerQNN", "Kernel", "kernel_matrix"]
