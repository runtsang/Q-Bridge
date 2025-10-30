"""Hybrid sampler‑estimator neural network.

This module defines a classical network that learns to output parameters
for a quantum sampler and a quantum estimator circuit. The network
consists of two sub‑networks: one that produces the 6 parameters
required by the sampler (2 inputs + 4 weights) and another that
produces the 2 parameters for the estimator (input + weight). Both
sub‑networks share a hidden representation to allow co‑adaptation
of sampler and estimator parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SamplerQNN(nn.Module):
    """Classical network that outputs parameters for a hybrid quantum sampler
    and estimator. The network takes a 2‑dimensional input and produces
    eight parameters: 2 sampler inputs, 4 sampler weights, 1 estimator
    input and 1 estimator weight. The hidden representation is shared
    between the two branches to enable joint optimisation."""
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # sampler branch
        self.sampler_head = nn.Linear(hidden_dim, 6)
        # estimator branch
        self.estimator_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dictionary with keys'sampler_params' and
        'estimator_params'."""
        h = self.trunk(x)
        sampler_params = self.sampler_head(h)
        estimator_params = self.estimator_head(h)
        return {
            "sampler_params": sampler_params,   # shape (..., 6)
            "estimator_params": estimator_params,  # shape (..., 2)
        }


__all__ = ["SamplerQNN"]
