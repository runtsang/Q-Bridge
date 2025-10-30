"""Enhanced fully‑connected layer for classical machine learning.

The class exposes a trainable linear transform with a tanh activation.
It can be used as a drop‑in replacement for the original FCL while
providing a lightweight training loop and gradient evaluation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn, optim


class FCL(nn.Module):
    """Trainable fully‑connected layer.

    Parameters
    ----------
    n_features : int, default 1
        Number of input features.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.activation(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the layer for a given set of parameters.

        Parameters
        ----------
        thetas : iterable of float
            Parameters to temporarily overwrite the linear weights.
            The length must match ``n_features``.
        """
        with torch.no_grad():
            weight = torch.tensor(list(thetas), dtype=torch.float32).view(1, -1)
            bias = torch.zeros(1, dtype=torch.float32)
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
            # Single‑sample forward
            out = self.forward(torch.zeros(1, len(thetas)))
            return out.detach().numpy()

    def train(
        self,
        x: Sequence[Sequence[float]],
        y: Sequence[float],
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """Simple MSE training loop.

        Parameters
        ----------
        x : sequence of feature vectors
        y : sequence of targets
        epochs : int
        lr : float
        verbose : bool
            If True, print loss every 20 epochs.
        """
        device = torch.device("cpu")
        X = torch.tensor(x, dtype=torch.float32, device=device)
        Y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            pred = self.forward(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        X = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy()

__all__ = ["FCL"]
