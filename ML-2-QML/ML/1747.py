"""Enhanced sampler neural network with deeper architecture and regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN__gen335(nn.Module):
    """
    A richer neural sampler that maps a 2‑dimensional input to a probability
    distribution over two outcomes. The architecture includes two hidden
    layers, batch normalisation, dropout and a soft‑max output.
    """
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability vector over two classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

def SamplerQNN():
    """Factory returning an instance of the enriched sampler."""
    return SamplerQNN__gen335()

__all__ = ["SamplerQNN", "SamplerQNN__gen335"]
