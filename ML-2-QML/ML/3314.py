"""Enhanced classical kernel and fully connected layer implementation.

The module builds upon the original RBF kernel and fully connected
layer by exposing a single cohesive class `QuantumKernelMethod`
that can be used as a drop‑in replacement for the legacy `Kernel`
and `KernalAnsatz`.  The kernel now supports a trainable `gamma`
parameter and broadcasting between arbitrary batch shapes, while
the fully‑connected sub‑module offers a simple linear layer with
tanh activation for quick prototyping of classification tasks.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable

class RBFKernel(nn.Module):
    """Classical radial basis function kernel with optional trainable gamma."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma)) if trainable else torch.tensor(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute exp(-gamma * ||x - y||^2) with broadcasting."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (len(x), len(y), d)
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

class FullyConnectedLayer(nn.Module):
    """Simple linear layer with tanh activation used for quick classification."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the linear layer on a sequence of scalars and return the mean tanh."""
        values = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(-1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

class QuantumKernelMethod(nn.Module):
    """
    Combined classical kernel and fully‑connected layer.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width; can be made trainable by passing ``trainable=True``.
    n_features : int, optional
        Input dimensionality for the fully‑connected layer.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 trainable_gamma: bool = False,
                 n_features: int = 1) -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma, trainable=trainable_gamma)
        self.fcl = FullyConnectedLayer(n_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value between two batches."""
        return self.kernel(x, y)

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of tensors."""
        K = torch.stack([self.forward(a_i, torch.stack(b)) for a_i in a])
        return K.detach().cpu().numpy()

    def fcl_run(self, thetas: Iterable[float]) -> np.ndarray:
        """Expose the fully‑connected layer's run method."""
        return self.fcl.run(thetas)

__all__ = ["QuantumKernelMethod", "RBFKernel", "FullyConnectedLayer"]
