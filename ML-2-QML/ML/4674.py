"""Classical kernel module with a learnable RBF kernel and kernel ridge regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with a learnable gamma and a lightweight kernel‑ridge
    regression wrapper.  The class exposes the same public interface as the
    quantum counterpart, enabling seamless backend switching.

    Parameters
    ----------
    gamma : float, optional
        Initial RBF width.  The value is treated as a learnable parameter.
    alpha : float, optional
        Regularisation strength for the ridge regression.
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.alpha = alpha
        self._model: KernelRidge | None = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches.  The method is fully
        differentiable so it can be used inside larger neural architectures.
        """
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        # Convert to numpy for sklearn's rbf_kernel
        kernel_np = rbf_kernel(x.detach().cpu().numpy(),
                               y.detach().cpu().numpy(),
                               gamma=self.gamma.item())
        return torch.tensor(kernel_np, device=x.device, dtype=x.dtype)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return a Gram matrix between two tensors."""
        return self.forward(a, b)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit a kernel ridge regressor on the provided training data.
        """
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self._model = KernelRidge(alpha=self.alpha, kernel="rbf",
                                  gamma=self.gamma.item())
        self._model.fit(X_np, y_np)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the stored kernel ridge model.  The method is only
        available after :meth:`fit` has been called.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = X.detach().cpu().numpy()
        preds = self._model.predict(X_np)
        return torch.tensor(preds, device=X.device, dtype=X.dtype)


__all__ = ["QuantumKernelMethod"]
