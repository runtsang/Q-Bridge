"""Extended feed-forward regressor with optional regularization and diagnostics.

The module implements the ExtendedEstimatorQNN class, a PyTorch neural network
designed to mirror the original EstimatorQNN while adding:
- Two hidden layers with ReLU activations.
- Dropout support for improved generalisation.
- A convenience ``predict`` method that accepts NumPy arrays.
- A ``train_on`` method that performs a single epoch of training given a
  PyTorch DataLoader, enabling quick experimentation.

The implementation intentionally keeps the API minimal so it can be dropped
into existing pipelines that expect a callable ``forward`` method.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class ExtendedEstimatorQNN(nn.Module):
    """A lightweight 2‑hidden‑layer regressor."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int, int] = (16, 8),
        dropout: float | None = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        dims = (input_dim, *hidden_dims, 1)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.pop()  # remove the last activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return predictions as a NumPy array."""
        self.eval()
        with torch.no_grad():
            inp = torch.as_tensor(X, dtype=torch.float32)
            out = self.forward(inp)
        return out.squeeze().numpy()

    def train_on(
        self,
        loader: DataLoader,
        lr: float = 1e-3,
        epochs: int = 1,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """Perform a quick training loop."""
        self.train()
        loss_fn = loss_fn or nn.MSELoss()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

__all__ = ["ExtendedEstimatorQNN"]
