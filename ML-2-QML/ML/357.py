"""Enhanced classical estimator with regularization and deeper architecture."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["EstimatorQNNGen"]


class EstimatorQNNGen(nn.Module):
    """
    A robust regression network that extends the original tiny feed‑forward model.
    Architecture:
        - Input: 2 features
        - Hidden layers: 16 → 32 units
        - Normalization: BatchNorm1d after each hidden layer
        - Activation: ReLU
        - Regularization: Dropout (p=0.2) after each hidden layer
        - Output: 1 regression value
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 32]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Predicted continuous value of shape (batch_size, 1).
        """
        return self.net(x)

    def predict(self, x: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Convenience method for evaluation.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        device : torch.device or str
            Device to perform computation on.

        Returns
        -------
        torch.Tensor
            Predicted values.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x.to(device))
