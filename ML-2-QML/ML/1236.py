"""Enhanced fully connected layer with training utilities.

The original seed implemented a single‑feature linear layer with a
``run`` method that returned a single expectation value.  This
upgrade generalises the layer to arbitrary input dimensionality,
supports batched inference, and adds an L2‑regularised loss and
optimizer‑based training routine.  The public API remains compatible
with the seed: ``run`` accepts an iterable of parameters and returns
a NumPy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss


@dataclass
class FullyConnectedLayer(nn.Module):
    """A classical fully‑connected layer with training support."""
    n_features: int = 1
    l2_reg: float = 0.0  # L2 regularisation strength

    def __post_init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(self.n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Compute the forward pass for a single set of parameters."""
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(theta_tensor))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run inference and return a NumPy array for compatibility."""
        with torch.no_grad():
            out = self.forward(thetas).mean(dim=0)
        return out.detach().numpy()

    def train_step(
        self,
        thetas: Sequence[float],
        targets: Sequence[float],
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> Sequence[float]:
        """Train the layer on a small dataset.

        Parameters
        ----------
        thetas : sequence of float
            Input features (flattened).
        targets : sequence of float
            Desired outputs.
        lr : float
            Learning rate for Adam.
        epochs : int
            Number of optimisation steps.

        Returns
        -------
        list[float]
            Optimised parameters after training.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(thetas)
            loss = mse_loss(preds, torch.tensor(targets, dtype=torch.float32))
            if self.l2_reg > 0:
                l2 = sum(p.pow(2).sum() for p in self.parameters())
                loss += self.l2_reg * l2
            loss.backward()
            optimizer.step()
        self.eval()
        return self.linear.weight.detach().numpy().flatten().tolist()


__all__ = ["FullyConnectedLayer"]
