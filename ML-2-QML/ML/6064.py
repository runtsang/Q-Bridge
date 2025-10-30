import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz preserving interface compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that reshapes inputs to 2‑D and delegates to :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute Gram matrix for two collections of 1‑D feature tensors."""
    kernel = Kernel(gamma)
    return torch.tensor([[kernel(x, y).item() for y in b] for x in a], dtype=torch.float32)

class EstimatorQNN__gen484(nn.Module):
    """
    Classical kernel‑ridge regressor that mirrors the EstimatorQNN example.
    It uses a radial‑basis function kernel and solves the regularised normal equations.
    """
    def __init__(self, gamma: float = 1.0, lambda_reg: float = 1e-3) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.alpha: torch.Tensor | None = None
        self.X_train: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the model by solving (K + λI)α = y.
        Parameters
        ----------
        X : torch.Tensor
            Training features of shape (n_samples, n_features).
        y : torch.Tensor
            Target values of shape (n_samples,).
        """
        K = kernel_matrix(X, X, gamma=self.gamma)
        n = K.size(0)
        I = torch.eye(n, dtype=K.dtype, device=K.device)
        A = K + self.lambda_reg * I
        self.alpha = torch.linalg.solve(A, y)
        self.X_train = X.clone()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict target values for new samples.
        Parameters
        ----------
        X : torch.Tensor
            Test features of shape (m_samples, n_features).
        Returns
        -------
        torch.Tensor
            Predicted values of shape (m_samples,).
        """
        if self.alpha is None or self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_test = kernel_matrix(X, self.X_train, gamma=self.gamma)
        return K_test @ self.alpha

__all__ = ["EstimatorQNN__gen484"]
