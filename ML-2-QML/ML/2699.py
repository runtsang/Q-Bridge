"""Hybrid classical neural network providing regression and categorical outputs.

This network extends the original EstimatorQNN by adding a softmax head for sampling,
mirroring the SamplerQNN functionality.  The shared trunk allows knowledge transfer
between the two tasks, improving sample efficiency and convergence speed."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridEstimatorSamplerQNN(nn.Module):
    """Two‑headed neural network providing regression and categorical outputs."""
    def __init__(self) -> None:
        super().__init__()
        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
        )
        # Regression head
        self.regressor = nn.Linear(8, 1)
        # Sampling head
        self.sampler = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (regression_output, softmax_distribution)."""
        h = self.trunk(x)
        reg = self.regressor(h)
        samp = F.softmax(self.sampler(h), dim=-1)
        return reg, samp

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression prediction."""
        return self.forward(x)[0]

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return categorical probabilities."""
        return self.forward(x)[1]

__all__ = ["HybridEstimatorSamplerQNN"]
