"""Enhanced fully connected layer with trainable weights and bias.

The class can be used as a drop‑in replacement for the original stand‑in.
It exposes a ``forward`` method, a ``loss`` method, and can be
integrated into a larger PyTorch model.  The layer supports
automatic differentiation and can be trained with any optimiser.
"""

import torch
from torch import nn
import numpy as np

class FCL(nn.Module):
    """Trainable fully connected layer with a single output neuron.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features.
    bias : bool, default=True
        Whether to include a bias term.
    """

    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.tanh(self.linear(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts NumPy arrays."""
        with torch.no_grad():
            return self.forward(torch.as_tensor(x, dtype=torch.float32)).numpy()

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,
             loss_fn: nn.Module = nn.MSELoss()) -> torch.Tensor:
        """Compute loss between predictions and ground truth."""
        return loss_fn(y_pred, y_true)
