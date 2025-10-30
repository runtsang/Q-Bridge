"""Hybrid classical sampler and estimator network.

The architecture combines a lightweight sampler (softmax output) and a regression head
using shared input features.  The sampler head mirrors the original SamplerQNN
while the estimator head extends the EstimatorQNN regression architecture.
The two heads share the first linear layer to reduce parameters and allow
joint training.

The module returns a tuple (probs, value) where probs is a probability
distribution over two classes and value is a scalar regression output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerEstimatorQNN(nn.Module):
    """
    Classical hybrid network with a sampler and an estimator head.

    Parameters
    ----------
    shared_dim : int, optional
        Size of the shared hidden layer.  Defaults to 8.
    """

    def __init__(self, shared_dim: int = 8) -> None:
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(2, shared_dim),
            nn.Tanh(),
        )
        # Sampler head: 2 -> 4 -> 2 -> softmax
        self.sampler_head = nn.Sequential(
            nn.Linear(shared_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Estimator head: 2 -> 8 -> 4 -> 1
        self.estimator_head = nn.Sequential(
            nn.Linear(shared_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing both sampler probabilities and regression output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        probs : torch.Tensor
            Softmax probabilities of shape (..., 2).
        value : torch.Tensor
            Regression output of shape (..., 1).
        """
        features = self.shared(inputs)
        probs = F.softmax(self.sampler_head(features), dim=-1)
        value = self.estimator_head(features)
        return probs, value


__all__ = ["HybridSamplerEstimatorQNN"]
