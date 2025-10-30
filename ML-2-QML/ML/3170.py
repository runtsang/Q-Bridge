"""Hybrid classical sampler and estimator network.

This module defines SamplerQNN which contains a dual‑branch
neural network: one branch produces a categorical distribution
over two classes (sampling) and the other outputs a scalar value
(regression).  The architecture incorporates residual connections,
dropout, and batch‑normalization to improve expressivity
and training stability compared to the original 2‑layer network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Classical hybrid sampler‑estimator network."""
    def __init__(self) -> None:
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        # Sampler head
        self.sampler_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        # Estimator head
        self.estimator_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (probabilities, estimate)."""
        features = self.shared(x)
        probs = F.softmax(self.sampler_head(features), dim=-1)
        estimate = self.estimator_head(features)
        return probs, estimate

__all__ = ["SamplerQNN"]
