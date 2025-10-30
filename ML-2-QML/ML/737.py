"""Enhanced fully connected layer with training utilities.

The class can be configured with an arbitrary number of hidden layers,
supports dropout and batch‑normalization, and exposes a convenient
``train_step`` method that performs a forward, loss computation and
back‑propagation.  The public API mirrors the original seed while
offering richer functionality.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class FullyConnectedLayer(nn.Module):
    """Multi‑layer perceptron with optional dropout and batch‑norm.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the input features.
    hidden_dims: Sequence[int], optional
        Sizes of the hidden layers.  Default: ``[64, 32]``.
    output_dim: int, optional
        Size of the output layer.  Default: ``1``.
    dropout: float, optional
        Dropout probability applied after each hidden layer.  Default: ``0.1``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Convenience helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return predictions for a 1‑D input array ``thetas``."""
        with torch.no_grad():
            x = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(-1)
            out = self.forward(x)
        return out.squeeze(-1).cpu().numpy()

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: nn.Module = nn.MSELoss(),
        optimizer: optim.Optimizer | None = None,
        lr: float = 1e-3,
        batch_size: int | None = None,
    ) -> float:
        """Perform one gradient step and return the loss value."""
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        if batch_size is None:
            batch_size = len(x)

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train()
        optimizer.zero_grad()
        preds = self.forward(x_tensor)
        loss = loss_fn(preds.squeeze(-1), y_tensor)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_params(self) -> dict[str, np.ndarray]:
        """Return a dictionary of the model parameters as NumPy arrays."""
        return {name: p.detach().cpu().numpy() for name, p in self.named_parameters()}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Set model parameters from a dictionary of NumPy arrays."""
        for name, arr in params.items():
            if name in self.state_dict():
                self.state_dict()[name].copy_(torch.as_tensor(arr))


def FCL(input_dim: int = 1) -> FullyConnectedLayer:
    """Return a fully‑connected layer instance.

    The default configuration matches the original seed but can be
    overridden by passing a custom ``input_dim``.
    """
    return FullyConnectedLayer(input_dim)


__all__ = ["FCL"]
