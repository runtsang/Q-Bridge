"""Enhanced classical fully‑connected layer with optional regularisation."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def FCL() -> nn.Module:
    """Return a fully‑connected layer with optional dropout and batch‑norm."""
    class FullyConnectedLayer(nn.Module):
        def __init__(
            self,
            n_features: int = 1,
            dropout_prob: float = 0.0,
            use_batchnorm: bool = False,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)
            self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
            self.bn = nn.BatchNorm1d(1) if use_batchnorm else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear(x)
            x = self.dropout(x)
            x = self.bn(x)
            return torch.tanh(x)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Compute the mean activation for a batch of input thetas."""
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            out = self.forward(values)
            expectation = out.mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


__all__ = ["FCL"]
