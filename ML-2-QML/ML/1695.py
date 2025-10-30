"""
SamplerQNNAdvanced – Classical implementation
================================================

This module provides a more expressive neural network for sampling tasks.
It adds batch‑norm, dropout and an extra hidden layer compared to the
original seed, improving expressivity while preserving the softmax
output.

Usage
-----
>>> from SamplerQNNAdvanced import SamplerQNNAdvanced
>>> model = SamplerQNNAdvanced()
>>> probs = model.sample(torch.tensor([0.5, -0.3]))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNAdvanced(nn.Module):
    """
    A robust sampler network with two hidden layers, batch‑norm and dropout.
    The network maps a 2‑dimensional input to a 2‑dimensional probability
    distribution via a softmax.  The architecture is suitable for use
    as the classical partner of a variational quantum sampler.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_dims : list[int], default [8, 4]
        Sizes of the two hidden layers.
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
        for idx, h_dim in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return the softmax probability distribution."""
        return F.softmax(self.net(x), dim=-1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a probability distribution for a given input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Probability tensor of shape (batch, 2).
        """
        return self.forward(x)


__all__ = ["SamplerQNNAdvanced"]
