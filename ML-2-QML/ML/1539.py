"""Enhanced classical fully‑connected layer with training capability."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    A multi‑layer perceptron that accepts a flat weight vector via ``run``.
    The architecture is configurable through ``hidden_dims`` and ``dropout``.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dims: tuple[int,...] = (10, 10),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Weight manipulation utilities
    # ------------------------------------------------------------------
    def _flatten_params(self) -> torch.Tensor:
        """Return all weight and bias parameters as a flat vector."""
        return torch.cat([p.view(-1) for p in self.parameters()])

    def _set_params_from_flat(self, flat: np.ndarray | torch.Tensor) -> None:
        """Assign a flat vector to all learnable parameters."""
        if isinstance(flat, np.ndarray):
            flat = torch.tensor(flat, dtype=torch.float32)
        ptr = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[ptr : ptr + numel].view_as(p))
            ptr += numel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, thetas: np.ndarray | Iterable[float]) -> np.ndarray:
        """
        Treat ``thetas`` as the input features, set the network weights from
        the flattened vector and evaluate the network mean output.
        """
        thetas = np.asarray(thetas, dtype=np.float32).flatten()
        # If the provided vector matches the expected weight size, set it.
        if len(thetas) == len(self._flatten_params()):
            self._set_params_from_flat(thetas)
        # Use the thetas as a dummy input to the network.
        inputs = torch.tensor(thetas, dtype=torch.float32).unsqueeze(1)
        out = self.model(inputs)
        return out.mean().detach().numpy()

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """
        Simple MSE training loop using Adam.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()


def FCL() -> FullyConnectedLayer:
    """Return a configurable fully‑connected neural network."""
    return FullyConnectedLayer()


__all__ = ["FullyConnectedLayer", "FCL"]
