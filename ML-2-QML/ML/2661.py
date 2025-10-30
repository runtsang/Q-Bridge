"""Hybrid sampler and estimator neural network.

This class merges the capabilities of the original SamplerQNN and EstimatorQNN
into a single torch.nn.Module.  The sampler sub‑network outputs a probability
distribution via softmax, while the estimator sub‑network outputs a scalar
prediction.  Both can be trained jointly, allowing experiments that exploit
shared feature representations for classification and regression tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQNN(nn.Module):
    """A hybrid sampler / estimator network.

    Parameters
    ----------
    sampler_hidden : int, optional
        Size of the hidden layer for the sampler branch.
    estimator_hidden : int, optional
        Size of the hidden layer for the estimator branch.
    """

    def __init__(self, sampler_hidden: int = 4, estimator_hidden: int = 8) -> None:
        super().__init__()
        # Sampler branch: 2 → sampler_hidden → 2 (softmax output)
        self.sampler_net = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )
        # Estimator branch: 2 → estimator_hidden → 4 → 1
        self.estimator_net = nn.Sequential(
            nn.Linear(2, estimator_hidden),
            nn.Tanh(),
            nn.Linear(estimator_hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing both sampler distribution and estimator value.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (sampler_output, estimator_output)
            sampler_output : softmax probabilities of shape (..., 2)
            estimator_output : regression output of shape (..., 1)
        """
        sampler_logits = self.sampler_net(x)
        sampler_output = F.softmax(sampler_logits, dim=-1)
        estimator_output = self.estimator_net(x)
        return sampler_output, estimator_output

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the sampler
        branch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2).
        n_samples : int, optional
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (..., n_samples).
        """
        probs, _ = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).transpose(0, 1)

__all__ = ["HybridQNN"]
