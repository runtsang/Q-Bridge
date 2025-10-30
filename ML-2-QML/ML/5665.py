"""Hybrid kernel module with classical and quantum back‑ends.

This module extends the original RBF kernel implementation by adding
a learnable gamma parameter, a wrapper that can automatically select a
classical or quantum kernel, and a differentiable kernel_matrix
function.  The API remains compatible with the original seed.

Classes
-------
KernalAnsatz
    Classical RBF kernel with learnable gamma.
Kernel
    Wrapper around KernalAnsatz for API compatibility.
HybridKernel
    Unified kernel module that can switch between classical and quantum
    back‑ends.  The quantum backend is optional and requires the QML
    module; attempting to use it without the QML implementation raises
    a clear error.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel with a learnable gamma.

    The gamma parameter is exposed as a :class:`torch.nn.Parameter`
    so it can be optimized jointly with a loss function.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Compatibility wrapper around :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

class HybridKernel(nn.Module):
    """Unified kernel module that can use a classical or quantum backend.

    Parameters
    ----------
    backend : {'classical', 'quantum'}
        Which backend to use. The default 'classical' requires no
        external dependencies.  The 'quantum' backend relies on the
        QML module and will raise ``ImportError`` if it is not
        available.
    gamma : float, optional
        Initial gamma value for the classical RBF kernel.
    """

    def __init__(self, backend: str = "classical", gamma: float = 1.0) -> None:
        super().__init__()
        self.backend = backend
        if backend == "classical":
            self.kernel = Kernel(gamma)
        elif backend == "quantum":
            try:
                from.qml_code import Kernel as QuantumKernel
            except Exception as exc:
                raise ImportError(
                    "Quantum backend requires the QML module to be importable."
                ) from exc
            self.kernel = QuantumKernel()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    backend: str = "classical",
) -> torch.Tensor:
    """Compute a differentiable Gram matrix between two datasets.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature vectors (each of shape ``(n_features,)``).
    backend : str, optional
        Which kernel backend to use. Default is 'classical'.

    Returns
    -------
    torch.Tensor
        Gram matrix of shape ``(len(a), len(b))``.
    """
    a = [x.unsqueeze(0) for x in a]
    b = [y.unsqueeze(0) for y in b]
    mat = torch.empty((len(a), len(b)), dtype=torch.float32)
    hk = HybridKernel(backend=backend, gamma=1.0)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            mat[i, j] = hk(xi, yj)
    return mat

__all__ = ["KernalAnsatz", "Kernel", "HybridKernel", "kernel_matrix"]
