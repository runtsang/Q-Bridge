"""Enhanced classical sampler network with regularisation and expanded capacity."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNExtended(nn.Module):
    """
    A two‑layer neural sampler with batch normalisation, dropout and a larger hidden dimension.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_dim : int, default 8
        Size of the hidden layer.
    output_dim : int, default 2
        Number of output classes (probabilities).
    dropout : float, default 0.1
        Drop‑out probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute a probability distribution over the output classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Soft‑max probabilities of shape (..., output_dim).
        """
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNNExtended() -> SamplerQNNExtended:
    """Return an instance of the enhanced sampler network."""
    return SamplerQNNExtended()


__all__ = ["SamplerQNNExtended"]
