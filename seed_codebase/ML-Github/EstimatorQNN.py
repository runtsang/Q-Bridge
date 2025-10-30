"""Tiny feed-forward regressor that mirrors the EstimatorQNN example."""

from __future__ import annotations

import torch
from torch import nn


def EstimatorQNN():
    """Return a simple fully-connected regression network."""

    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
