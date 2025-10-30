"""Hybrid fully‑connected layer with optional neural‑network backbone.

This module extends the original single‑layer example by adding a small
feed‑forward network that processes the input parameters before they are
passed to the tanh activation.  The resulting class is a standard
PyTorch `nn.Module`, so it can be trained with any optimiser.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Tuple


def FCL(
    n_features: int = 1,
    hidden_dim: int = 32,
    depth: int = 1,
    device: str = "cpu",
) -> nn.Module:
    """Return a fully‑connected module that can be used in a training loop.

    Parameters
    ----------
    n_features:
        Number of input features per sample.
    hidden_dim:
        Size of the hidden layer.
    depth:
        Number of hidden layers.
    device:
        Target device for the tensors.
    """

    class FullyConnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = [nn.Linear(n_features, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)
            self.to(device)

        def forward(self, thetas: Iterable[float]) -> torch.Tensor:
            """Forward pass returning a single scalar per sample."""
            x = torch.as_tensor(list(thetas), dtype=torch.float32, device=device)
            return torch.tanh(self.net(x))

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Convenience wrapper that mimics the original API."""
            with torch.no_grad():
                val = self.forward(thetas).mean()
            return val.cpu().numpy()

        def train_step(
            self,
            thetas: Iterable[float],
            target: float,
            loss_fn: nn.Module = nn.MSELoss(),
            optimizer: nn.Module = None,
        ) -> float:
            """Perform a single optimisation step."""
            if optimizer is None:
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            optimizer.zero_grad()
            pred = self.forward(thetas).mean()
            loss = loss_fn(pred, torch.tensor(target, device=self.net[0].weight.device))
            loss.backward()
            optimizer.step()
            return loss.item()

    return FullyConnectedLayer()


__all__ = ["FCL"]
