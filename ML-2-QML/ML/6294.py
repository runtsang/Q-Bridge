"""
QuantumKernelMethod (classical implementation).

Provides a classical RBF kernel, a kernel‑matrix routine,
and a lightweight neural sampler network.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KernalAnsatz(nn.Module):
    """
    Simple RBF kernel ansatz.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """
    Wrapper that ensures the input tensors are 1‑D.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class SamplerQNN(nn.Module):
    """
    Tiny feed‑forward sampler that outputs a probability vector.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class QuantumKernelMethod:
    """
    Public API that exposes the classical kernel and sampler.
    """
    def __init__(self, gamma: float = 1.0, device: str = "cpu") -> None:
        self.kernel = Kernel(gamma)
        self.sampler = SamplerQNN()
        self.device = device

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of 1‑D tensors.
        """
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the sampler on the provided inputs.
        """
        return self.sampler(inputs.to(self.device))
    
__all__ = ["KernalAnsatz", "Kernel", "SamplerQNN", "QuantumKernelMethod"]
