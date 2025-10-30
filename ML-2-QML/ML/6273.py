"""Hybrid classical kernel method with optional neural sampler.

The class `QuantumKernelMethod` implements a classical radial basis
function kernel augmented by a lightweight neural sampler.  The sampler
produces a probability distribution over two classes that is used to
weight the kernel value, providing a simple form of data‑dependent
re‑weighting.  This design demonstrates how a purely classical
implementation can be enriched with a quantum‑inspired architecture
while remaining fully compatible with the original API."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Simple two‑layer neural sampler returning a softmax over two
    outputs.  It mirrors the quantum SamplerQNN but stays fully
    classical."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class KernalAnsatz(nn.Module):
    """Classical RBF kernel component."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumKernelMethod:
    """Hybrid kernel that optionally uses a neural sampler to re‑weight
    the classical RBF kernel.  The API matches the original
    `QuantumKernelMethod` class so that downstream code continues to
    work unchanged."""
    def __init__(self, gamma: float = 1.0, use_sampler: bool = False):
        self.kernel = Kernel(gamma)
        self.use_sampler = use_sampler
        self.sampler = SamplerQNN() if use_sampler else None

    def _sample_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampler is None:
            return torch.ones(1, device=x.device)
        probs = self.sampler(x)
        return probs[:, 0].unsqueeze(-1)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.kernel(x, y)
        weight = self._sample_weight(x)
        return k * weight

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.sampler is None:
            raise RuntimeError("Sampler not enabled for this instance.")
        return self.sampler(inputs)


__all__ = ["SamplerQNN", "KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
