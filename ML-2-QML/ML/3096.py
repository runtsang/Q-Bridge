"""Hybrid QNN combining sampling and estimation networks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQNN(nn.Module):
    """
    Classical neural network that first samples a probability distribution via a
    small feed‑forward network and then regresses a target value from that distribution.
    The architecture merges the original SamplerQNN and EstimatorQNN designs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sampler stage – 2‑to‑4‑to‑2 softmax
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Estimator stage – 2‑to‑8‑to‑4‑to‑1 regression
        self.estimator = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass that produces a probability vector and a scalar output.
        Parameters
        ----------
        inputs
            Tensor of shape (batch, 2) containing classical features.
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 3) where the first two columns are the
            probabilities from the sampler and the last column is the regression
            prediction.
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        pred = self.estimator(probs)
        return torch.cat([probs, pred], dim=-1)


__all__ = ["HybridQNN"]
