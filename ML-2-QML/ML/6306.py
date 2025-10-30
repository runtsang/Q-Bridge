"""Enhanced classical sampler network with deeper architecture and regularisation.

The original SamplerQNN implemented a shallow 2‑to‑2 feed‑forward network.
This extension introduces additional hidden units, batch‑normalisation,
and dropout to improve generalisation.  The module can be used as a drop‑in
replacement for the original and is fully compatible with PyTorch training
pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """Deep sampler network with dropout and batch‑norm."""

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        input_dim:
            Dimensionality of the input feature vector.
        hidden_dims:
            Sequence of hidden layer sizes.  Defaults to [8, 4].
        """
        super().__init__()
        hidden_dims = hidden_dims or [8, 4]
        layers: list[nn.Module] = []

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, input_dim))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return class‑probability distribution via softmax."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


def SamplerQNN() -> SamplerModule:
    """Convenience factory mirroring the original interface."""
    return SamplerModule()


__all__ = ["SamplerQNN", "SamplerModule"]
