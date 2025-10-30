"""Enhanced classical fully‑connected layer with trainable MLP and dropout.

The class keeps the original lightweight API but adds depth, regularisation
and a simple training helper.  It can be dropped into existing pipelines
without changing the surrounding code.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
from torch import nn


class EnhancedFCL(nn.Module):
    """
    Multi‑layer perceptron that mimics the original `FCL` but with
    hidden layers, ReLU activations and dropout.

    Parameters
    ----------
    n_features : int
        Number of input features (default 1).
    hidden_sizes : Sequence[int]
        Sizes of hidden layers (default [16, 8]).
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Iterable[int] = (16, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass for compatibility with the original API.

        Parameters
        ----------
        thetas : Iterable[float]
            Input values treated as a batch of single‑feature samples.

        Returns
        -------
        np.ndarray
            Mean output over the batch as a 1‑element array.
        """
        # Convert to a column vector: (batch, 1)
        x = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(1)
        out = self.forward(x).mean(dim=0)
        return out.detach().cpu().numpy()

    def train_step(
        self,
        thetas: Iterable[float],
        targets: Iterable[float],
        lr: float = 1e-3,
        loss_fn: nn.Module | None = None,
    ) -> float:
        """
        One training step for quick prototyping.

        Parameters
        ----------
        thetas : Iterable[float]
            Batch of input features.
        targets : Iterable[float]
            Ground‑truth values.
        lr : float, optional
            Learning rate for the optimizer.
        loss_fn : nn.Module, optional
            Loss function; defaults to MSELoss.

        Returns
        -------
        float
            Loss value for the step.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = loss_fn or nn.MSELoss()

        optimizer.zero_grad()
        preds = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(1)
        preds = self.forward(preds).squeeze()
        loss = loss_fn(preds, torch.tensor(list(targets), dtype=torch.float32))
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["EnhancedFCL"]
