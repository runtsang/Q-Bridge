"""Enhanced fully connected layer with training utilities.

The original seed exposed a single linear layer wrapped in a ``run`` method.
This extension adds:
- Arbitrary depth (``n_hidden``) with optional dropout.
- A ``forward`` method to integrate with PyTorch training loops.
- A helper ``train_step`` that performs one gradient descent update.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn, optim
import numpy as np


class FCL(nn.Module):
    """
    Multi‑layer fully connected network with dropout.

    Parameters
    ----------
    n_features : int
        Size of the input feature vector.
    n_hidden : Sequence[int], optional
        Sizes of hidden layers.  If omitted a single linear layer is used.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """

    def __init__(self,
                 n_features: int = 1,
                 n_hidden: Sequence[int] | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        layers: list[nn.Module] = []

        # Input layer
        in_dim = n_features
        if n_hidden:
            for out_dim in n_hidden:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.Tanh())
                layers.append(self.dropout)
                in_dim = out_dim
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters to feed into the network.
        Returns
        -------
        torch.Tensor
            Output tensor of shape (1,).
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return self.net(x).view(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Convenience wrapper returning a NumPy array, mirroring the seed."""
        return self.forward(thetas).detach().numpy()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_step(self,
                   thetas: Iterable[float],
                   target: float,
                   lr: float = 1e-3,
                   loss_fn: nn.Module | None = None) -> float:
        """
        Perform one gradient‑descent step on the given data point.

        Parameters
        ----------
        thetas : Iterable[float]
            Current parameters.
        target : float
            Desired output.
        lr : float, optional
            Learning rate.
        loss_fn : nn.Module, optional
            Loss function; defaults to MSELoss.

        Returns
        -------
        float
            Loss value after the step.
        """
        self.train()
        loss_fn = loss_fn or nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        optimizer.zero_grad()
        pred = self.forward(thetas)
        loss = loss_fn(pred, torch.tensor([target], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FCL"]
