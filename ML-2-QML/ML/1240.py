"""Enhanced classical sampler network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A two‑layer neural sampler with residual connections, dropout, and batch‑norm.
    The network maps a 2‑dimensional input to a probability distribution over 2 classes.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input vector.
        hidden_dim : int
            Size of the hidden linear layer.
        dropout : float
            Dropout probability applied after the hidden layer.
        seed : int | None
            Random seed for reproducibility.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.residual = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution over the output classes.
        """
        out = self.net(inputs)
        out = out + self.residual(inputs)  # residual connection
        return F.softmax(out, dim=-1)


def SamplerQNNFactory(**kwargs) -> SamplerQNN:
    """
    Factory that returns a SamplerQNN instance.
    The function name mirrors the legacy API to preserve compatibility.
    """
    return SamplerQNN(**kwargs)


__all__ = ["SamplerQNNFactory", "SamplerQNN"]
