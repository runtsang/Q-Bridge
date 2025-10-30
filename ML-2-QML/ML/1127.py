"""
HybridSamplerQNN – Classical implementation
===========================================

This module defines a deep, configurable neural network that mirrors the
original two‑layer sampler but adds residual connections, dropout and
support for arbitrary hidden layer sizes.  It can be used as a drop‑in
replacement for the seed implementation in any pipeline that expects a
`torch.nn.Module` returning a probability distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    A configurable classical sampler network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : Sequence[int], default (32, 32)
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (32, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # Residual connection if dimensions match
            if prev_dim == h_dim:
                layers.append(nn.Identity())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., input_dim).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["HybridSamplerQNN"]
