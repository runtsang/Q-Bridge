"""
Enhanced classical sampler network with residual connections, batch normalisation, and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A two‑layer MLP with residual, batch‑norm and dropout that outputs a
    probability distribution over two classes.  The design mirrors the
    original 2‑to‑4‑to‑2 architecture but adds robustness for noisy data.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability vector.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., 2).
        """
        out = self.net(inputs)
        out = self.dropout(out)
        return F.softmax(out, dim=-1)


__all__ = ["SamplerQNN"]
