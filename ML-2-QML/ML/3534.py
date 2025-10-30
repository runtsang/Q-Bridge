"""Hybrid EstimatorQNN combining classical self‑attention and feed‑forward regression.

The network first applies a learnable self‑attention module to the input, producing an
attended representation that is then fed through a small fully‑connected regressor.
All parameters are jointly optimised during training.
"""

import torch
from torch import nn
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Trainable self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the attention embedding.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable rotation parameters (3 per output dimension)
        self.rotation_params = nn.Parameter(
            torch.randn(embed_dim * 3, dtype=torch.float32)
        )
        # Learnable entanglement parameters (1 per pair of dimensions)
        self.entangle_params = nn.Parameter(
            torch.randn(embed_dim - 1, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute attended representations.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Attended tensor of shape (batch, embed_dim).
        """
        query = inputs @ self.rotation_params.reshape(
            self.embed_dim, -1
        )
        key = inputs @ self.entangle_params.reshape(
            self.embed_dim, -1
        )
        value = inputs
        scores = torch.softmax(
            query @ key.T / np.sqrt(self.embed_dim), dim=-1
        )
        return scores @ value


class FeedForwardRegressor(nn.Module):
    """Simple feed‑forward regressor used after attention."""
    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridEstimatorQNN(nn.Module):
    """Hybrid estimator combining attention and regression."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.regressor = FeedForwardRegressor(input_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        return self.regressor(attended)


def EstimatorQNN() -> HybridEstimatorQNN:
    """
    Return a fresh instance of the hybrid EstimatorQNN.

    Returns
    -------
    HybridEstimatorQNN
        Initialized model ready for training.
    """
    return HybridEstimatorQNN()


__all__ = ["EstimatorQNN"]
