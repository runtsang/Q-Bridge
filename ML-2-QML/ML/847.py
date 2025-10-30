"""Hybrid classical kernel module with trainable bandwidth and kernel‑ridge regression wrapper."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with a learnable bandwidth (gamma) and a kernel‑ridge regression wrapper.
    The class can be trained end‑to‑end using gradient descent on the bandwidth and the regression
    coefficients.
    """

    def __init__(self, gamma: float = 1.0, lambda_reg: float = 1e-5, lr: float = 1e-3, epochs: int = 200):
        """
        Parameters
        ----------
        gamma : float, optional
            Initial bandwidth of the RBF kernel.
        lambda_reg : float, optional
            Regularisation strength for the ridge regression.
        lr : float, optional
            Learning rate for the optimiser.
        epochs : int, optional
            Number of optimisation iterations.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.alpha = None
        self.X_train = None
        self.y_train = None

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix using the RBF kernel.
        """
        # Ensure shapes: (n, d), (m, d)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (n, m, d)
        sq_norm = torch.sum(diff ** 2, dim=2)   # (n, m)
        return torch.exp(-self.gamma * sq_norm)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the kernel matrix.
        """
        return self.kernel_matrix(X, Y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelMethod":
        """
        Fit the kernel ridge regression model.
        Optimises the bandwidth (gamma) and computes the regression coefficients.
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.X_train = X
        self.y_train = y

        optimizer = torch.optim.Adam([self.gamma], lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            K = self.kernel_matrix(X, X)
            K_reg = K + self.lambda_reg * torch.eye(K.shape[0], device=K.device)
            alpha = torch.linalg.solve(K_reg, y)
            y_pred = K @ alpha
            loss = torch.mean((y_pred - y) ** 2)
            loss.backward()
            optimizer.step()

        # Store the final coefficients
        K = self.kernel_matrix(X, X)
        K_reg = K + self.lambda_reg * torch.eye(K.shape[0], device=K.device)
        self.alpha = torch.linalg.solve(K_reg, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        X = torch.tensor(X, dtype=torch.float32)
        K_test = self.kernel_matrix(X, self.X_train)
        return (K_test @ self.alpha).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
