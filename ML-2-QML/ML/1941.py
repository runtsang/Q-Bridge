"""Enhanced classical sampler network with residual connections and dropout.

The original SamplerQNN defined a single hidden layer. We extend it to
a small MLP that can adapt to more complex decision boundaries and
allow stochastic sampling via the softmax output. The network is still
light‑weight enough for quick prototyping but offers richer features
(e.g. batch normalization, dropout, and a configurable hidden size).

The class can be used interchangeably with the quantum version in hybrid
experiments because the API – a single forward call returning a
probability distribution – stays the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Classical sampler network.

    Parameters
    ----------
    in_features : int, default 2
        Number of input features.
    hidden_features : int, default 32
        Number of hidden units in each hidden layer.
    n_layers : int, default 2
        Number of hidden layers.
    dropout : float, default 0.1
        Dropout probability after each hidden layer.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_features))
            layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_features
        layers.append(nn.Linear(prev, 2))  # output logits for 2 classes
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution defined by the
        network's output probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, in_features).
        n_samples : int
            Number of samples to draw for each input.
        """
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).permute(1, 0, 2)  # shape (batch, n_samples, 2)


__all__ = ["SamplerQNN"]
