"""Classical stand-in for the fully connected quantum layer example."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def FCL():
    """Return an object with a ``run`` method mimicking the quantum example."""

    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


__all__ = ["FCL"]
