"""
EstimatorQNNEnhanced: A flexible classical feed‑forward regressor.
The class accepts a layer specification, activation function, and dropout ratio, allowing
experiments with depth, width, and regularisation.  The public API mirrors the seed
EstinatorQNN, returning a callable model ready for training with PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence, Callable, Optional


class EstimatorQNNEnhanced(nn.Module):
    """
    A configurable fully‑connected regression network.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Sequence of hidden layer sizes. The first element is the input dimension
        (default 2), the last element is the output dimension (default 1).
    activation : Callable[[torch.Tensor], torch.Tensor] | str, optional
        Activation function or its name. Default is 'tanh'.
    dropout : float, optional
        Dropout probability applied after each hidden layer. 0 disables dropout.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int] = (2, 8, 4, 1),
        activation: Callable[[torch.Tensor], torch.Tensor] | str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            act = getattr(nn, activation.capitalize())()
        else:
            act = activation

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def EstimatorQNNEnhancedModel() -> EstimatorQNNEnhanced:
    """Convenience factory returning a default‑configured model."""
    return EstimatorQNNEnhanced()


__all__ = ["EstimatorQNNEnhanced", "EstimatorQNNEnhancedModel"]
