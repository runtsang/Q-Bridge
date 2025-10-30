import numpy as np
import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    Extended feed‑forward regressor.
    • Accepts an arbitrary list of hidden layer sizes via *hidden_dims*.
    • Adds batch‑normalisation and dropout for regularisation.
    • Provides a ``predict`` helper that works with NumPy arrays.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (8, 4), dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.Tanh(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts a NumPy array."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X.astype(np.float32))
            return self.forward(X).cpu().numpy().squeeze()


__all__ = ["EstimatorQNN"]
