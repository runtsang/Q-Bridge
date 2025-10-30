"""SamplerQNNGen312 – classical deep sampler."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, BatchNorm1d

__all__ = ["SamplerQNNGen312"]


class SamplerQNNGen312(nn.Module):
    """
    A two‑hidden‑layer neural sampler with batch‑norm and dropout.
    The network maps a 2‑dimensional input to a 2‑dimensional probability
    vector via a softmax.  It can be used as a drop‑in replacement for
    the original SamplerQNN while providing higher capacity and
    regularisation.

    Parameters
    ----------
    input_dim : int, default 2
        Dimension of the input vector.
    hidden_dims : list[int], default [8, 4]
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] = [8, 4],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
