"""Hybrid classical kernel module.

The module keeps the original API (``KernalAnsatz``, ``Kernel``,
``kernel_matrix``) while adding a weighted combination with a
user‑supplied quantum kernel.  It also offers a lightweight classical
QCNN analogue that mirrors the quantum design.

The code is pure NumPy/PyTorch and can run on CPU or GPU.
"""

from __future__ import annotations

from typing import Callable, Sequence, Optional

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Classic RBF kernel ansatz.

    Retains the same constructor signature as the anchor file for
    backward compatibility.  ``gamma`` controls the width of the
    Gaussian.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Ensure row‑major shape
        x = x if x.ndim > 1 else x.unsqueeze(0)
        y = y if y.ndim > 1 else y.unsqueeze(0)
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=2))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that matches the seed API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Return Gram matrix for classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class HybridKernel(nn.Module):
    """
    Combines a classical RBF kernel with a quantum kernel supplied
    as a callable.  The callable must accept two torch tensors and
    return a NumPy array of shape (n, m).  ``quantum_weight`` controls
    the mixing ratio.

    The hybrid kernel is useful for experiments that want to
    progressively augment a classical model with quantum features
    without rewriting the entire training loop.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 quantum_weight: float = 0.5,
                 quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], np.ndarray]] = None) -> None:
        super().__init__()
        self.classical = KernalAnsatz(gamma)
        self.quantum_weight = quantum_weight
        self.quantum_kernel = quantum_kernel

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        Kc = self.classical(x, y)
        if self.quantum_kernel is None:
            return Kc
        Kq = torch.from_numpy(self.quantum_kernel(x, y)).to(Kc.device, dtype=Kc.dtype)
        return (1 - self.quantum_weight) * Kc + self.quantum_weight * Kq


class QCNNModel(nn.Module):
    """
    Classical convolution‑inspired network that mimics the structure
    of the quantum QCNN.  It consists of a feature map and three
    convolution‑pooling stages followed by a classification head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning a ready‑to‑train :class:`QCNNModel`."""
    return QCNNModel()


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernel",
    "QCNNModel",
    "QCNN",
]
