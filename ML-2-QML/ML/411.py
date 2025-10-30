"""Upgraded fully connected layer with PyTorch MLP and dropout."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCLGen035(nn.Module):
    """
    A flexible multi‑layer perceptron that mimics a quantum fully‑connected layer.

    Parameters
    ----------
    n_features : int
        Size of the input feature vector.
    hidden_layers : Sequence[int], optional
        Number of units per hidden layer. If ``None`` a single linear layer is used.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        n_features: int = 1,
        hidden_layers: Sequence[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [1]
        layers: list[nn.Module] = []
        in_features = n_features
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        # Final output layer
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input values, one per sample.

        Returns
        -------
        np.ndarray
            Normalized output of the network, shape (1,).
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        y = self.network(x).mean(dim=0)
        return torch.tanh(y).detach().cpu().numpy()


def FCL() -> FCLGen035:
    """Return an instance of the upgraded fully‑connected layer."""
    return FCLGen035()


__all__ = ["FCL", "FCLGen035"]
