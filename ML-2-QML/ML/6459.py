"""Hybrid classical sampler & regressor network.

The backbone processes two input features and produces both a categorical
distribution (softmax) and a scalar regression output.  The architecture
combines the lightweight sampler from the original seed with the deeper
regressor from EstimatorQNN, enabling joint training of generative and
predictive objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerEstimatorNet(nn.Module):
    """Shared backbone with two heads: sampler and estimator."""
    def __init__(self) -> None:
        super().__init__()
        # shared layers
        self.backbone = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )
        # sampler head
        self.sampler_head = nn.Linear(8, 2)
        # estimator head
        self.estimator_head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dictionary with'sample' (softmax probs) and 'estimate'."""
        z = self.backbone(x)
        sample = F.softmax(self.sampler_head(z), dim=-1)
        estimate = self.estimator_head(z)
        return {"sample": sample, "estimate": estimate}

__all__ = ["SamplerEstimatorNet"]
