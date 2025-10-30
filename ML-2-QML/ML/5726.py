"""Enhanced classical sampler network with configurable hidden layers and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class SamplerModule(nn.Module):
    """A flexible feed‑forward sampler.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_layers : Sequence[int] | None, default None
        Sizes of hidden layers. If empty a single linear layer is used.
    dropout : float | None, default None
        Dropout probability applied after each non‑linear activation.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = []

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over the input dimension."""
        return F.softmax(self.net(inputs), dim=-1)

def SamplerQNN(**kwargs) -> SamplerModule:
    """Instantiate a SamplerModule with optional hyper‑parameters.

    Example
    -------
    >>> model = SamplerQNN(hidden_layers=[8, 8], dropout=0.1)
    """
    return SamplerModule(**kwargs)

__all__ = ["SamplerQNN"]
