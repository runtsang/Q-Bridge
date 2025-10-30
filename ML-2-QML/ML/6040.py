"""Hybrid classical estimator and sampler model.

This module defines HybridEstimatorSampler that combines a regression network
and a softmax sampler. The regression head predicts continuous weights for
a downstream quantum circuit, while the sampler head produces a probability
distribution over two outcomes.  The two heads are trained jointly, enabling
the classical part to drive both a quantum expectation value and a sampling
task.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridEstimatorSampler(nn.Module):
    """Combined regression and sampling network."""
    def __init__(self) -> None:
        super().__init__()
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Sampling head
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (regression_output, sampling_output)."""
        reg = self.regressor(inputs)          # shape (...,1)
        samp = F.softmax(self.sampler(inputs), dim=-1)  # shape (...,2)
        return reg, samp

__all__ = ["HybridEstimatorSampler"]
