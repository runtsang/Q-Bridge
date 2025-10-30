"""Hybrid classical estimator combining feed‑forward regression with a self‑attention module.

The network first maps the raw features into an attention space using learnable query and key
projections, then aggregates the values via a soft‑max weighted sum.  The resulting
representation is fed to a lightweight fully‑connected regressor.  This design preserves
the simplicity of the original EstimatorQNN while adding a contextual weighting
mechanism inspired by the SelfAttention reference.

The class is fully PyTorch‑compatible and can be used in standard training pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid estimator.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    embed_dim : int, default 4
        Dimensionality of the attention embedding.
    hidden_dims : list[int], default [8, 4]
        Sizes of the hidden layers in the regression head.
    """

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 4,
        hidden_dims: list[int] | tuple[int,...] = (8, 4),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Attention projections
        self.query_weight = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.key_weight = nn.Parameter(torch.randn(input_dim, embed_dim))

        # Regression head
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar value of shape (batch, 1).
        """
        # Self‑attention
        query = x @ self.query_weight          # (batch, embed_dim)
        key = x @ self.key_weight              # (batch, embed_dim)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_output = scores @ x               # (batch, input_dim)

        # Regression
        return self.net(attn_output)


__all__ = ["HybridEstimatorQNN"]
