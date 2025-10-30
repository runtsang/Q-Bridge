"""Enhanced classical sampler network.

This implementation extends the original two‑layer network by allowing
arbitrary input dimension and hidden depth, optional dropout and
batch‑normalization.  It can be used as a drop‑in replacement for the
original SamplerQNN helper in downstream pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A flexible sampler network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimension of the input feature vector.
    hidden_dims : Sequence[int], default (4,)
        Sizes of hidden linear layers.
    dropout : float, default 0.0
        Drop‑out probability; 0 disables dropout.
    batch_norm : bool, default False
        Whether to insert BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (4,),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over the input dimension."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
