"""
Classical sampler with a deep, regularised neural network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A robust classical sampler network.

    Architecture:
        - Input: 2‑dimensional feature vector (e.g. two‑dimensional data point).
        - Hidden: 8‑unit linear layers with BatchNorm, ReLU and Dropout.
        - Output: 2‑dimensional probability vector (softmax).

    The network is intentionally deeper and regularised to mitigate over‑fitting
    while still remaining lightweight for quick prototyping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 2).
        num_samples : int
            Number of samples to draw per batch element.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (batch, num_samples).
        """
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples, replacement=True)

__all__ = ["SamplerQNN"]
