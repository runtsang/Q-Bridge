from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Extended sampler network with deeper architecture and regularisation.
    Provides a probability distribution over 2 classes from a 2‑dimensional input.
    """
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a softmax probability vector.
        """
        logits = self.model(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
