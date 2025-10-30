"""Hybrid kernel and estimator for classical experiments."""

from __future__ import annotations

from typing import Sequence, Iterable
import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz with broadcasting support."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute pairwise squared Euclidean distances efficiently
        # x: (n, d), y: (m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        dist2 = torch.sum(diff * diff, dim=-1)   # (n, m)
        return torch.exp(-self.gamma * dist2)    # (n, m)


class Kernel(nn.Module):
    """Wraps KernalAnsatz and provides a callable kernel matrix."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between sequences of tensors using the RBF kernel."""
    x = torch.stack(a)
    y = torch.stack(b)
    kernel = Kernel(gamma)
    return kernel(x, y).cpu().numpy()


class EstimatorNN(nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self, input_dim: int = 2, hidden_dims: Iterable[int] = (8, 4), output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


def EstimatorQNN() -> EstimatorNN:
    """Return a classical estimator mirroring the QML example."""
    return EstimatorNN()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "EstimatorNN", "EstimatorQNN"]
