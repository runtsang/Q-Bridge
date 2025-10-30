from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class CombinedQNN(nn.Module):
    """
    A hybrid classical network that provides both regression and sampling outputs.
    The regression head is a small feed‑forward regressor; the sampling head
    produces a probability distribution via a softmax.
    """

    def __init__(self) -> None:
        super().__init__()
        # Regression sub‑network
        self.regressor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Sampling sub‑network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both regression output and sampling probabilities.
        """
        reg_out = self.regressor(x)
        samp_out = F.softmax(self.sampler(x), dim=-1)
        return reg_out, samp_out

__all__ = ["CombinedQNN"]
