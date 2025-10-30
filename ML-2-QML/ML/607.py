"""Enhanced classical RBF kernel utilities with vectorized operations and regression support."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class RBFKernel(nn.Module):
    """
    Radial basis function kernel with support for batched inputs and GPU acceleration.

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel width parameter.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for two batches of samples.

        Parameters
        ----------
        x : torch.Tensor
            Shape (n_samples_x, n_features).
        y : torch.Tensor
            Shape (n_samples_y, n_features).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n_samples_x, n_samples_y).
        """
        # Ensure 2D tensors
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])

        # Efficient broadcasted squared distance
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_dist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_dist)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of tensors using the RBF kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors representing samples.
    gamma : float, default=1.0
        Kernel width.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = RBFKernel(gamma)
    # Stack into a single tensor for batch computation
    a_stack = torch.stack(a)
    b_stack = torch.stack(b)
    return kernel(a_stack, b_stack).cpu().numpy()


class KernelRidgeRegression(nn.Module):
    """
    Simple kernel ridge regression using the RBF kernel.

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel width.
    alpha : float, default=1e-3
        Regularization strength.
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 1e-3) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self._alpha = alpha
        self._train_X = None
        self._train_y = None
        self._dual_coef = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "KernelRidgeRegression":
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : torch.Tensor
            Training features of shape (n_samples, n_features).
        y : torch.Tensor
            Target values of shape (n_samples,).
        """
        self._train_X = X
        self._train_y = y
        K = RBFKernel(self.gamma)(X, X)
        n = K.shape[0]
        # Closed‑form solution: (K + alpha * I)^{-1} y
        K_reg = K + self._alpha * torch.eye(n, device=K.device)
        self._dual_coef = torch.linalg.solve(K_reg, y)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict target values for new samples.

        Parameters
        ----------
        X : torch.Tensor
            Query features of shape (n_query, n_features).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (n_query,).
        """
        if self._train_X is None or self._dual_coef is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_test = RBFKernel(self.gamma)(X, self._train_X)
        return K_test @ self._dual_coef


__all__ = ["RBFKernel", "kernel_matrix", "KernelRidgeRegression"]
