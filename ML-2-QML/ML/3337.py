"""Hybrid classical sampler network.

The network outputs a soft‑max probability over two classes and a
vector of four trainable weight parameters that will be fed into the
quantum sampler.  The architecture is deliberately shallow to keep
training fast while still providing enough capacity for hybrid
experiments."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical component of the hybrid sampler.

    Attributes
    ----------
    encoder : nn.Sequential
        Maps the 2‑dimensional input to 4 weight parameters.
    prob_net : nn.Sequential
        Produces a 2‑dimensional probability distribution.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4,
                 weight_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, weight_dim),
        )
        self.prob_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 2)``.

        Returns
        -------
        probs : torch.Tensor
            Soft‑max probabilities of shape ``(batch, 2)``.
        weight_params : torch.Tensor
            Weight parameters for the quantum circuit,
            shape ``(batch, 4)``.
        """
        weight_params = self.encoder(x)
        probs = F.softmax(self.prob_net(x), dim=-1)
        return probs, weight_params

__all__ = ["HybridSamplerQNN"]
