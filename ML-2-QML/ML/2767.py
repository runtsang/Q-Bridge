"""Classical quantum‑kernel estimator using an RBF kernel and a linear regressor.

The class is intentionally lightweight: it accepts a gamma parameter for the radial basis
function, builds a kernel matrix, and trains a scikit‑learn regressor on that matrix.
The implementation mirrors the reference `QuantumKernelMethod` while extending it with
end‑to‑end fitting and prediction logic.

Typical usage:

    from QuantumKernelMethod__gen020 import QuantumKernelEstimator
    X_train, y_train =...
    model = QuantumKernelEstimator(gamma=0.5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
#  Classical RBF kernel utilities – adapted from the seed
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """Legacy RBF kernel component – kept for API compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that exposes a `forward` method compatible with the original seed."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#  High‑level estimator
# --------------------------------------------------------------------------- #

class QuantumKernelEstimator:
    """
    Train a regression model on a kernel matrix.

    Parameters
    ----------
    gamma : float
        Width of the RBF kernel.
    reg : sklearn‑like object
        Regressor that implements ``fit`` and ``predict``.  Defaults to
        :class:`sklearn.linear_model.LinearRegression`.

    Notes
    -----
    The estimator accepts raw data tensors; it internally converts them to
    a kernel matrix and delegates the learning task to the supplied regressor.
    """

    def __init__(self, gamma: float = 1.0, reg: LinearRegression | None = None) -> None:
        self.gamma = gamma
        self.reg = reg or LinearRegression()
        self._kernel = Kernel(gamma)

    # --------------------------------------------------------------------- #
    #  Kernel helpers
    # --------------------------------------------------------------------- #
    def _kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between X and Y."""
        return np.array([[self._kernel(x, y).item() for y in Y] for x in X])

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def fit(self, X: Sequence[torch.Tensor], y: Iterable[float]) -> None:
        """Fit the underlying regressor on the kernel matrix."""
        X = torch.stack(list(X))
        K = self._kernel_matrix(X, X)
        self.reg.fit(K, np.asarray(list(y)))

    def predict(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        """Predict on new data using the trained regressor."""
        X = torch.stack(list(X))
        K = self._kernel_matrix(X, X)
        return self.reg.predict(K)

    def __repr__(self) -> str:
        return f"<QuantumKernelEstimator gamma={self.gamma} reg={self.reg!r}>"

__all__ = ["QuantumKernelEstimator"]
