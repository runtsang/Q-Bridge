"""Hybrid classical network combining regression and sampling sub-networks."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridQNN(nn.Module):
    """
    Classical hybrid network that merges a regression head and a sampler head.
    The regression head predicts a scalar value, while the sampler head outputs
    a probability distribution over two classes.  This mirrors the structure
    of the EstimatorQNN and SamplerQNN seeds but allows joint forward passes.
    """

    def __init__(self) -> None:
        super().__init__()
        # Regression subnetwork (mirrors EstimatorQNN)
        self.estimator_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Sampler subnetwork (mirrors SamplerQNN)
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns a tuple (regression_output, sampler_probs).

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Regression output of shape (..., 1) and probability distribution
            of shape (..., 2).
        """
        reg_out = self.estimator_net(inputs)
        sampler_logits = self.sampler_net(inputs)
        sampler_probs = F.softmax(sampler_logits, dim=-1)
        return reg_out, sampler_probs

    def parameters(self, recurse: bool = True) -> torch.nn.ParameterList:
        """
        Expose all parameters from both sub-networks.
        """
        return super().parameters(recurse=recurse)


__all__ = ["HybridQNN"]
