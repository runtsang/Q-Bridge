# FCL__gen087.py – Classical implementation
"""A robust classical fully‑connected layer with training capability.

The original seed provided a toy layer that merely applied a linear map
followed by a tanh.  This extension introduces a two‑layer neural net,
a routine for loading flat parameter lists, and a simple MSE
optimiser.  The API is unchanged – ``FCL()`` returns an object with a
``run`` method – so downstream scripts that imported the seed continue
to work while gaining real learning functionality.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import torch
from torch import nn, optim


class FullyConnectedLayer(nn.Module):
    """Two‑layer neural network that mimics a quantum fully‑connected layer."""

    def __init__(self, n_features: int = 1, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def set_params_from_list(self, params: Sequence[float]) -> None:
        """Load flat list of parameters into the network (weights then biases)."""
        flat = torch.tensor(params, dtype=torch.float32)
        idx = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                w_shape = layer.weight.shape
                b_shape = layer.bias.shape
                w_num = np.prod(w_shape)
                b_num = np.prod(b_shape)
                layer.weight.data = flat[idx : idx + w_num].view(w_shape)
                idx += w_num
                layer.bias.data = flat[idx : idx + b_num].view(b_shape)
                idx += b_num

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Forward pass given a flat list of parameters."""
        self.set_params_from_list(thetas)
        with torch.no_grad():
            x = torch.as_tensor(thetas, dtype=torch.float32).view(1, -1)
            out = self.net(x)
        return out.squeeze().detach().numpy()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Train the network to minimise MSE on the supplied data."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        # Return the network’s output on the last parameter snapshot
        return self.run(self.parameters().data.flatten().tolist())


def FCL() -> FullyConnectedLayer:
    """Return a fully‑connected layer ready for ``run`` and ``train``."""
    return FullyConnectedLayer()


__all__ = ["FCL"]
