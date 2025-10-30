"""Extended classical sampler network with depth and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A deeper, regularized sampler network.

    Architecture:
        Input (2) -> Linear(4) -> Tanh
        -> Dropout(0.2)
        -> Linear(8) -> ReLU
        -> Dropout(0.2)
        -> Linear(2) -> Softmax
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution over 2 classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) representing two input features.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., 2).
        """
        return F.softmax(self.net(inputs), dim=-1)

    def get_params(self) -> dict:
        """Return a dictionary of named parameters for introspection."""
        return {name: param.detach().cpu().numpy() for name, param in self.named_parameters()}


__all__ = ["SamplerQNN"]
