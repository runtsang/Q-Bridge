"""Fully connected neural network with flexible depth and training utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


class FCL(nn.Module):
    """
    A flexible multilayer perceptron that mimics the behaviour of a quantum
    fully‑connected layer but with classical tensors.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dims : Sequence[int]
        Sizes of hidden layers; an empty sequence yields a single linear layer.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        layers: list[nn.Module] = []

        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum interface: accept a sequence of parameters,
        run a forward pass, and return the mean activation as a NumPy array.
        """
        with torch.no_grad():
            input_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            out = self.forward(input_tensor)
            return out.mean().item()

    def train_step(
        self,
        thetas: Iterable[float],
        target: float,
        lr: float = 1e-3,
        clip_norm: float | None = None,
    ) -> float:
        """
        Perform a single gradient‑descent step against a scalar target.
        Returns the loss value.
        """
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(
            torch.tensor(self.run(thetas), dtype=torch.float32), torch.tensor(target)
        )
        loss.backward()

        if clip_norm is not None:
            clip_grad_norm_(self.parameters(), clip_norm)

        optimizer.step()
        return loss.item()
