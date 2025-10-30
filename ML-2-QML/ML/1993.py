"""Enhanced classical fully connected layer with trainable parameters and simple training loop."""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """
    Fully connected layer with one hidden unit and tanh activation.
    Provides a `run` method for compatibility with the original FCL example and
    a `train_on` method that optimises the internal weights on a toy regression task.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.activation(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Legacy interface: accept an iterable of parameters, feed them
        through the linear layer and return the mean tanh output as a NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.activation(self.linear(values)).mean(dim=0)
        return out.detach().numpy()

    def train_on(self, X: np.ndarray, y: np.ndarray, epochs: int = 200,
                 lr: float = 1e-3, verbose: bool = False) -> None:
        """
        Very small training helper that optimises the layer against a mean‑squared‑error loss.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
            if verbose and epoch % max(1, (epochs // 5)) == 0:
                print(f"Epoch {epoch:04d} loss={loss.item():.4f}")

def FCL():
    """Return the FullyConnectedLayer class for backward compatibility."""
    return FullyConnectedLayer

__all__ = ["FullyConnectedLayer", "FCL"]
