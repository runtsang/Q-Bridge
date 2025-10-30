"""Hybrid classical estimator inspired by EstimatorQNN and QuantumKernelMethod.

The :class:`EstimatorQNNGen345` class is a pure‑Python surrogate that
combines a small feed‑forward network with a kernel‑ridge regressor.
It can use either a classical RBF kernel or a user‑supplied quantum
kernel (e.g. from the QML module).  The implementation is fully
compatible with scikit‑learn pipelines.
"""

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from typing import Callable, Optional

class EstimatorQNNGen345(BaseEstimator, RegressorMixin):
    """Classical surrogate for the quantum EstimatorQNN.

    Parameters
    ----------
    hidden_sizes : list[int]
        Sizes of hidden layers in the surrogate NN.
    lr : float
        Learning rate for the NN optimizer.
    epochs : int
        Number of training epochs.
    kernel : Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
        Kernel function to use in the KernelRidge regressor.
    alpha : float
        Regularization strength for KernelRidge.
    verbose : bool
        Whether to print training progress.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = (8, 4),
        lr: float = 1e-3,
        epochs: int = 200,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        alpha: float = 1e-2,
        verbose: bool = False,
    ):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.kernel = kernel
        self.alpha = alpha
        self.verbose = verbose
        self._net = None
        self._kr = None

    def _build_net(self, input_dim: int):
        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self._net = nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EstimatorQNNGen345":
        """Fit the surrogate NN and kernel ridge regressor."""
        self._build_net(X.shape[1])
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self._net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs} loss: {loss.item():.4f}")

        if self.kernel is None:
            self.kernel = (
                lambda a, b: np.exp(-np.linalg.norm(a[:, None] - b[None, :], axis=2) ** 2)
            )
        self._kr = KernelRidge(kernel=self.kernel, alpha=self.alpha)
        self._kr.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using a weighted blend of NN and kernel ridge."""
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            nn_pred = self._net(X_t).squeeze().numpy()
        kr_pred = self._kr.predict(X)
        return 0.7 * nn_pred + 0.3 * kr_pred

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(hidden_sizes={self.hidden_sizes}, "
            f"lr={self.lr}, epochs={self.epochs})"
        )

__all__ = ["EstimatorQNNGen345"]
