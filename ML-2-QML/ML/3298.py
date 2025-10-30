"""Hybrid classical kernel model with sampler network.

The module mirrors the quantum implementation in order to
allow direct comparison.  It exposes a single
`HybridKernelModel` class that can be instantiated with a
`gamma` parameter for the RBF kernel and an optional
`sampler` flag to switch on the sampler network.  The class
provides a `kernel_matrix` method that returns a Gram matrix
between two collections of data points and a `predict`
method that applies a simple linear classifier on the
kernel features.

Typical usage:

    model = HybridKernelModel(gamma=0.5, use_sampler=True)
    K = model.kernel_matrix(X, X_test)
    y_pred = model.predict(K)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Optional

__all__ = ["HybridKernelModel"]


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that accepts
    batched inputs and returns a scalar kernel value.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class SamplerQNN(nn.Module):
    """Simple linear sampler network that maps a 2‑D input
    to a probability distribution over two outcomes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a soft‑max distribution."""
        return F.softmax(self.net(inputs), dim=-1)


class HybridKernelModel(nn.Module):
    """Hybrid kernel model that optionally augments the
    classical RBF kernel with a sampler network.

    Parameters
    ----------
    gamma : float
        Width of the RBF kernel.
    use_sampler : bool
        If ``True`` the sampler network is applied to each
        data point before computing the kernel.
    """

    def __init__(self, gamma: float = 1.0, use_sampler: bool = False) -> None:
        super().__init__()
        self.kernel = Kernel(gamma)
        self.use_sampler = use_sampler
        self.sampler: Optional[SamplerQNN] = SamplerQNN() if use_sampler else None
        # A trivial linear classifier that takes the kernel
        # features as input.  It is kept separate so that the
        # model can be trained with ``torch.optim`` if desired.
        self.classifier = nn.Linear(1, 1, bias=False)

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the sampler (if enabled) to each sample."""
        if self.use_sampler and self.sampler is not None:
            return self.sampler(data)
        return data

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between ``a`` and ``b``."""
        a_t = torch.stack([self._transform(x) for x in a])
        b_t = torch.stack([self._transform(y) for y in b])
        return np.array(
            [[self.kernel(a_t[i], b_t[j]).item() for j in range(len(b_t))] for i in range(len(a_t))]
        )

    def predict(self, kernel_mat: np.ndarray) -> np.ndarray:
        """Apply the linear classifier to the kernel matrix."""
        # Convert to torch tensor for the classifier
        k = torch.from_numpy(kernel_mat).float()
        preds = self.classifier(k.unsqueeze(-1)).squeeze(-1)
        return preds.detach().numpy()
