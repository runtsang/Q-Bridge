"""Hybrid classical kernel model with estimator and sampler."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""

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
    """Compute the Gram matrix between two batches of samples."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class EstimatorNN(nn.Module):
    """Simple feed‑forward regression network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SamplerModule(nn.Module):
    """Probabilistic output network for sampling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridKernelModel:
    """Unified interface for classical kernel, estimator, and sampler."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.kernel = Kernel(gamma)
        self.estimator = EstimatorNN()
        self.sampler = SamplerModule()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, self.kernel.ansatz.gamma)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.estimator(x)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return self.sampler(x)


__all__ = ["HybridKernelModel"]
