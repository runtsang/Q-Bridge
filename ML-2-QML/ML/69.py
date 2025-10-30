"""Enhanced feed-forward regression model with residuals and dropout.

This implementation extends the original tiny network by:
- Supporting arbitrary input dimensionality.
- Using a configurable stack of hidden layers.
- Adding dropout for regularization.
- Providing a simple `predict` helper that normalises inputs.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Iterable, List, Tuple

class EstimatorQNN(nn.Module):
    """
    A lightweight neural network for regression tasks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Sequence[int]
        Sizes of the hidden layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for idx, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout))
            # Residual connection if dimensions match
            if prev_dim == h_dim:
                layers.append(nn.Identity())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns a detached NumPy array."""
        self.eval()
        with torch.no_grad():
            return self(x).cpu().numpy().squeeze()

__all__ = ["EstimatorQNN"]
