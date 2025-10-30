"""Enhanced fully connected layer with training capability.

The original seed implemented a single linear layer with a tanh
activation and a ``run`` method that accepted a list of parameters.
This upgraded version turns the layer into a small multi‑layer
perceptron, adds dropout and batch‑normalisation, and exposes a
``train`` helper that optimises the network on a given dataset.
The public API remains compatible: ``run`` still accepts an
iterable of parameters and returns a NumPy array of predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FCL(nn.Module):
    """Multi‑layer perceptron with optional dropout and batch‑norm.

    Parameters
    ----------
    n_features : int
        Size of the input feature vector.
    hidden_sizes : list[int] | tuple[int] | None, optional
        Sizes of the hidden layers.  Defaults to an empty list (single
        linear output).
    dropout : float, optional
        Drop‑out probability applied after every hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: list[int] | tuple[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: list[float] | np.ndarray) -> np.ndarray:
        """Run the network with a supplied feature vector.

        Parameters
        ----------
        thetas : iterable of float
            Input feature vector of length ``n_features``.
        """
        x = torch.tensor(thetas, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            out = self.forward(x)
        return np.array([out.item()])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """Mini‑batch SGD training on the supplied data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples, 1)
            Target vector.
        epochs : int, default 10
            Number of training epochs.
        lr : float, default 1e-3
            Learning rate.
        batch_size : int, default 32
            Mini‑batch size.
        """
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()


__all__ = ["FCL"]
