"""
FCLayer – Classical fully‑connected layer with advanced training utilities.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCLayer(nn.Module):
    """
    A flexible, stackable fully connected layer that can be used as a drop‑in replacement
    for the original single‑parameter demo.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    hidden : Sequence[int], optional
        Sizes of intermediate hidden layers. If ``None`` the module reduces to a single
        linear layer.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.
    bias : bool, default=True
        Whether to include bias terms in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Sequence[int] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = in_features

        # Build hidden layers if requested
        if hidden:
            for h in hidden:
                layers.append(nn.Linear(last_dim, h, bias=bias))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                last_dim = h

        # Final output layer
        layers.append(nn.Linear(last_dim, out_features, bias=bias))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. ``x`` is expected to be of shape ``(batch, in_features)``.
        Returns a tensor of shape ``(batch, out_features)``.
        """
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original API: accept a list of scalars, feed them through the network,
        and return the mean activation as a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input values. They are reshaped to a column vector and passed
            through the network.

        Returns
        -------
        np.ndarray
            Mean output over the batch dimension.
        """
        input_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(input_tensor)
        return out.mean(dim=0).detach().numpy()

    def train_step(
        self,
        thetas: Iterable[float],
        targets: Iterable[float],
        lr: float = 1e-3,
        loss_fn: nn.Module | None = None,
    ) -> float:
        """
        Perform one gradient‑descent update on the network.

        Parameters
        ----------
        thetas : Iterable[float]
            Input data.
        targets : Iterable[float]
            Ground‑truth targets.
        lr : float
            Learning rate.
        loss_fn : nn.Module, optional
            Loss function. If ``None`` defaults to MSELoss.

        Returns
        -------
        float
            Current loss value.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        optimizer.zero_grad()
        preds = self.forward(
            torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        )
        loss = loss_fn(preds.squeeze(), torch.as_tensor(list(targets), dtype=torch.float32))
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FCLayer"]
