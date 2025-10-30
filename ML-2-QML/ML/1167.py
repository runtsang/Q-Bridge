"""Classical RBF kernel with hyperparameter tuning and kernel ridge regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel module with optional hyperparameter optimization.
    Provides fit/predict interface for regression tasks using Kernel Ridge.
    """

    def __init__(self, gamma: float | None = None, alpha: float = 1.0, cv_folds: int = 5):
        """
        Parameters
        ----------
        gamma : float | None
            Width of the RBF kernel. If None, a logarithmic grid will be searched.
        alpha : float
            Regularization strength for Kernel Ridge.
        cv_folds : int
            Number of cross‑validation folds for hyperparameter search.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.model: KernelRidge | None = None

    def rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value between two input samples."""
        return self.rbf(x, y).squeeze()

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        X_np, Y_np = X.numpy(), Y.numpy()
        return np.exp(-self.gamma * np.sum((X_np[:, None, :] - Y_np[None, :, :]) ** 2, axis=-1))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit a Kernel Ridge model to the data."""
        X_np, y_np = X.numpy(), y.numpy()
        if self.gamma is None:
            # search over a logarithmic grid
            param_grid = {"gamma": np.logspace(-3, 3, 7)}
            grid = GridSearchCV(KernelRidge(alpha=self.alpha), param_grid, cv=self.cv_folds)
            grid.fit(X_np, y_np)
            self.gamma = grid.best_params_["gamma"]
            self.model = grid.best_estimator_
        else:
            self.model = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)
            self.model.fit(X_np, y_np)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict target values for new samples."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return torch.from_numpy(self.model.predict(X.numpy()))

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Return mean squared error on the given test set."""
        y_pred = self.predict(X)
        return mean_squared_error(y.numpy(), y_pred.numpy())

__all__ = ["QuantumKernelMethod"]
