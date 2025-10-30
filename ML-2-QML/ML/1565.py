"""
Enhanced classical SamplerQNN implementation.

Features:
- Configurable number of hidden layers and units.
- Optional dropout and batch‑normalisation.
- Soft‑max output suitable for probability distributions.
- Utility method to compute KL‑divergence against a target distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence


class SamplerQNN(nn.Module):
    """
    A flexible neural sampler.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dims : Sequence[int]
        Sizes of hidden layers. Default: (8, 8).
    dropout : float | None
        Dropout probability. If None, dropout is omitted.
    batch_norm : bool
        Whether to insert BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        dropout: float | None = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or (8, 8)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the input dimension."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def kl_divergence(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL‑divergence KL(probs ‖ target).

        Parameters
        ----------
        probs : torch.Tensor
            Predicted probabilities (batch, dim).
        target : torch.Tensor
            Target probabilities (batch, dim).

        Returns
        -------
        torch.Tensor
            KL‑divergence for each sample in the batch.
        """
        eps = 1e-12
        probs = torch.clamp(probs, eps, 1.0)
        target = torch.clamp(target, eps, 1.0)
        return torch.sum(target * (torch.log(target) - torch.log(probs)), dim=-1)


__all__ = ["SamplerQNN"]
