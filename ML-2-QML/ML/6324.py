"""Enhanced classical regressor with configurable depth and dropout regularization.

The network depth is defined by a sequence of hidden layer sizes, each followed by a
Tanh activation and an optional Dropout.  The public API mirrors the original
seed: EstimatorQNN() returns a nn.Module instance and accepts the same keyword
arguments as EstimatorNN.__init__.
"""

import torch
from torch import nn


class EstimatorNN(nn.Module):
    """Feed‑forward regression network with configurable depth and dropout."""

    def __init__(self, hidden_sizes: list[int] | tuple[int,...] | None = None,
                 dropout: float = 0.0) -> None:
        """
        Parameters
        ----------
        hidden_sizes:
            Sequence of hidden layer sizes.  If ``None`` defaults to ``[8, 4]``.
        dropout:
            Dropout probability applied after every hidden layer.  ``0.0`` disables dropout.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 4]
        layers = []
        in_features = 2
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


def EstimatorQNN(**kwargs) -> EstimatorNN:
    """Factory that returns a configured EstimatorNN instance.

    The function accepts the same keyword arguments as EstimatorNN.__init__.
    """
    return EstimatorNN(**kwargs)


__all__ = ["EstimatorNN", "EstimatorQNN"]
