"""Hybrid classical surrogate for the quantum sampler‑estimator.

The architecture augments the original 2‑layer SamplerQNN with a deeper
encoder and dual heads, enabling it to approximate both the probability
distribution and the expectation value produced by the 192‑parameter quantum
circuit.  The network is fully PyTorch‑compatible and can be used for
pre‑training or as a baseline for comparison with the quantum model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerEstimator(nn.Module):
    """
    Classical surrogate model that predicts both a probability vector and a
    scalar regression output, mimicking the hybrid quantum sampler‑estimator.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder that processes the 2‑dimensional input
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )
        # Two heads: one for distribution (softmax) and one for regression
        self.prob_head = nn.Linear(16, 2)
        self.reg_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass produces a probability vector and a scalar estimate.
        Args:
            x: Tensor of shape (..., 2)
        Returns:
            probs: softmaxed probabilities of shape (..., 2)
            reg: regression output of shape (..., 1)
        """
        h = self.encoder(x)
        probs = F.softmax(self.prob_head(h), dim=-1)
        reg = self.reg_head(h)
        return probs, reg


__all__ = ["HybridSamplerEstimator"]
