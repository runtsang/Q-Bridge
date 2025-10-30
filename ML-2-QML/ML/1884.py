"""Enhanced classical sampler network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSamplerQNN(nn.Module):
    """
    A feed‑forward network that maps a 2‑dimensional input to a 2‑dimensional probability vector.
    The architecture can be extended with an optional hidden layer and dropout.
    """

    def __init__(self, hidden_dim: int = 8, dropout: float = 0.0) -> None:
        """
        Parameters
        ----------
        hidden_dim : int
            Size of the optional intermediate hidden layer.
        dropout : float
            Dropout probability applied after the hidden layer.
        """
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution via softmax.
        """
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNN() -> AdvancedSamplerQNN:
    """
    Factory returning an instance of :class:`AdvancedSamplerQNN`.
    """
    return AdvancedSamplerQNN()


__all__ = ["AdvancedSamplerQNN", "SamplerQNN"]
