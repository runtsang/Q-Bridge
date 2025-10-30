"""Advanced classical sampler network extending the original architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSamplerModule(nn.Module):
    """
    A richer sampler network using batch normalization, dropout, and deeper layers.
    Mirrors the original 2→4→2 structure but expands capacity for complex distributions.
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
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over the two output classes."""
        return F.softmax(self.net(inputs), dim=-1)

def SamplerQNN() -> AdvancedSamplerModule:
    """
    Factory returning an instance of the advanced sampler module.
    """
    return AdvancedSamplerModule()

__all__ = ["SamplerQNN"]
