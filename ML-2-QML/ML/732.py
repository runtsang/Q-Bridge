"""Enhanced feed-forward regressor with dropout and batch normalization.

Provides a flexible architecture that can be easily tuned for regression tasks.
"""

import torch
from torch import nn
from typing import Sequence, Tuple

class EstimatorQNN(nn.Module):
    """
    A versatile fully‑connected neural network for regression.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    output_dim : int
        Size of the output layer (default 1 for scalar regression).
    dropout : float
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool
        If True, inserts a BatchNorm1d after each hidden layer.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: Sequence[int] = (16, 8),
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    @staticmethod
    def synthetic_data(n_samples: int = 1024,
                       noise: float = 0.1,
                       seed: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a toy regression dataset.

        Returns
        -------
        X : torch.Tensor
            Feature matrix of shape (n_samples, 2).
        y : torch.Tensor
            Target vector of shape (n_samples, 1).
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        X = torch.randn(n_samples, 2, generator=rng)
        y = (X[:, 0] ** 2 - X[:, 1]).unsqueeze(-1) + noise * torch.randn(n_samples, 1, generator=rng)
        return X, y

__all__ = ["EstimatorQNN"]
