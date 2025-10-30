"""Enhanced classical regressor with modular architecture.

The network supports arbitrary input size, multiple hidden layers,
dropout, batch‑normalisation, and a convenient ``train`` helper
that runs a single epoch and returns the loss value.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import optim

__all__ = ["EstimatorQNN"]


class EstimatorQNN(nn.Module):
    """Configurable feed‑forward regressor.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the input feature vector.
    hidden_dims: Sequence[int]
        List of hidden layer sizes.  An empty list yields a linear
        mapping from ``input_dim`` to the output.
    dropout: float, default 0.0
        Drop‑out probability applied after each hidden layer.
    activation: nn.Module, default nn.ReLU
        Activation function used after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int,...] | None = None,
        *,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    activation,
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(inputs)

    @staticmethod
    def default() -> "EstimatorQNN":
        """Return a sensible default architecture."""
        return EstimatorQNN(input_dim=2, hidden_dims=[8, 4], dropout=0.1)

    def train_epoch(
        self,
        loader,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ) -> float:
        """Run a single training epoch over ``loader`` and return the loss.

        Parameters
        ----------
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
            Data loader yielding (inputs, targets).
        lr: float
            Learning rate.
        weight_decay: float
            L2 regularisation.
        device: str
            Target device.
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        self.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = self(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        return total_loss / len(loader.dataset)
