"""Classical kernel module with exact RBF and optional random Fourier feature approximation.

The :class:`QuantumKernelMethod` class extends the original ``Kernel`` by:
* supporting an exact RBF kernel or an RFF‑approximated kernel via the ``kernel_type`` argument;
* lazily constructing the random feature matrix when the first pair of vectors is evaluated;
* exposing both a standard ``kernel_matrix`` and an ``approximate_kernel_matrix`` that uses the RFF feature map.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Optional

class QuantumKernelMethod(nn.Module):
    """Flexible classical kernel supporting exact RBF and random Fourier feature (RFF) approximation.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF width parameter.
    kernel_type : str, default='rbf'
        Either ``'rbf'`` for the exact kernel or ``'rff'`` for the approximation.
    n_features : int, optional
        Dimensionality of the random Fourier feature map. Required if ``kernel_type='rff'``.
    random_state : int, optional
        Seed for reproducible random feature generation.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        kernel_type: str = "rbf",
        n_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.kernel_type = kernel_type.lower()
        self.n_features = n_features
        self.random_state = random_state

        if self.kernel_type not in {"rbf", "rff"}:
            raise ValueError(
                f"Unsupported kernel_type: {self.kernel_type}. Use 'rbf' or 'rff'."
            )

        if self.kernel_type == "rff" and self.n_features is None:
            raise ValueError(
                "n_features must be specified when using the 'rff' kernel_type."
            )

        # Random features will be constructed lazily on first use.
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact RBF kernel."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _rff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel approximated via random Fourier features."""
        if self.W is None:
            rng = np.random.default_rng(self.random_state)
            dim = x.shape[-1]
            self.W = rng.normal(
                0, np.sqrt(2 * self.gamma), size=(self.n_features, dim)
            )
            self.b = rng.uniform(0, 2 * np.pi, size=self.n_features)

        phi_x = np.sqrt(2 / self.n_features) * np.cos(
            np.dot(self.W, x.numpy()) + self.b
        )
        phi_y = np.sqrt(2 / self.n_features) * np.cos(
            np.dot(self.W, y.numpy()) + self.b
        )
        return torch.tensor(np.dot(phi_x, phi_y)).unsqueeze(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel between two 1‑D vectors."""
        if self.kernel_type == "rbf":
            return self._rbf(x, y)
        else:
            return self._rff(x, y)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the Gram matrix between two datasets."""
        a = torch.stack(a)
        b = torch.stack(b)
        return np.array(
            [
                [self.forward(x, y).item() for y in b]
                for x in a
            ]
        )

    def approximate_kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return the kernel matrix using the RFF approximation."""
        if self.kernel_type!= "rff":
            raise RuntimeError(
                "Approximate kernel matrix is only available for 'rff' kernel_type."
            )
        a = torch.stack(a)
        b = torch.stack(b)
        return np.array(
            [
                [self.forward(x, y).item() for y in b]
                for x in a
            ]
        )

__all__ = ["QuantumKernelMethod"]
