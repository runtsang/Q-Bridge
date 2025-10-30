"""Hybrid classical sampler‑estimator network.

This module defines a single PyTorch module that combines the
probabilistic sampler of SamplerQNN with the regression head of
EstimatorQNN.  The sampler network outputs a probability vector
over two classes; this vector is used as a feature representation
for the estimator head, which produces a scalar prediction.
The design keeps the classical network fully differentiable
and compatible with standard training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQNN(nn.Module):
    """A hybrid classical sampler‑estimator network.

    Attributes
    ----------
    sampler_net : nn.Sequential
        Produces a 2‑dimensional probability distribution.
    estimator_net : nn.Sequential
        Takes the sampler probabilities as input and outputs a scalar.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sampler part – identical to the original SamplerQNN
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Estimator part – identical to the original EstimatorQNN
        self.estimator_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampler probabilities and estimator prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 2) – raw input features.

        Returns
        -------
        probs : torch.Tensor
            Shape (batch, 2) – softmax probabilities from the sampler.
        preds : torch.Tensor
            Shape (batch, 1) – regression output from the estimator.
        """
        probs = F.softmax(self.sampler_net(inputs), dim=-1)
        preds = self.estimator_net(probs)
        return probs, preds


__all__ = ["HybridQNN"]
