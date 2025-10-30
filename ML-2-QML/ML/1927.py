"""Enhanced classical fully‑connected layer with optional depth and dropout."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


def FCL() -> nn.Module:
    """Return a configurable fully‑connected neural network.

    The network accepts a 1‑D input vector ``thetas`` and outputs a single
    scalar prediction.  It supports multiple hidden layers, ReLU
    activations and optional dropout, making it suitable for regression
    or classification tasks that benefit from deeper representations.
    """

    class FullyConnectedLayer(nn.Module):
        """Parameterized feed‑forward network."""

        def __init__(
            self,
            n_features: int = 1,
            hidden_sizes: Sequence[int] = (32,),
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            layers = []
            prev = n_features
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.model = nn.Sequential(*layers)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Forward pass using the provided input vector.

            Parameters
            ----------
            thetas : Iterable[float]
                Input feature vector.  Length must match ``n_features``.

            Returns
            -------
            np.ndarray
                Predicted scalar value.
            """
            x = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self.model(x)
            return out.squeeze().detach().numpy()

    return FullyConnectedLayer()


__all__ = ["FCL"]
