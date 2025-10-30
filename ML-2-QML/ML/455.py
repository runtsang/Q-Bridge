"""
Classical sampler network with enhanced expressivity and sampling utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A richer, regularised neural sampler.

    Features
    --------
    * Three dense layers with batch‑normalisation and dropout.
    * Log‑softmax output for stable probability estimation.
    * ``sample`` helper that draws discrete samples from the predicted distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass returning log‑probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(..., 2)`` representing the two‑dimensional input.

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape ``(..., 2)``.
        """
        return F.log_softmax(self.net(inputs), dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(..., 2)``.
        num_samples : int
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., num_samples)`` containing sampled indices.
        """
        log_probs = self.forward(inputs)
        probs = log_probs.exp()
        return torch.multinomial(probs, num_samples=num_samples, replacement=True)


__all__ = ["SamplerQNN"]
