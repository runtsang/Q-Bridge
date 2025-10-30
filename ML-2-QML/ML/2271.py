"""HybridSamplerEstimatorQNN: Classical two‑head network for sampling and regression.

This module defines a PyTorch neural network that mirrors the
SamplerQNN and EstimatorQNN seeds, but combines them into a single
model with a shared backbone and two output heads. The sampler head
produces a probability distribution via softmax, while the estimator
head outputs a scalar regression value. The architecture is
parameter‑sized to allow easy scaling and joint training.

The design follows the combination scaling paradigm: the classical
model is a single module that can be trained end‑to‑end, and it
provides a clean API for both sampling and estimation tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerEstimatorQNN(nn.Module):
    """Two‑head feed‑forward network for sampling and regression."""

    def __init__(self) -> None:
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
        )
        # Sampler head: outputs 2 probabilities
        self.sampler_head = nn.Linear(8, 2)
        # Estimator head: outputs a single real value
        self.estimator_head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (probabilities, regression output)
        """
        features = self.backbone(x)
        probs = F.softmax(self.sampler_head(features), dim=-1)
        pred = self.estimator_head(features)
        return probs, pred


__all__ = ["HybridSamplerEstimatorQNN"]
