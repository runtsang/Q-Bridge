"""Hybrid fully‑connected layer – classical implementation.

The module mirrors the structure of the original FCL and SamplerQNN,
combining a two‑layer feed‑forward network with a softmax output.
It exposes a ``run`` method that accepts a sequence of parameters
(`thetas`) and returns a tuple ``(probs, expectation)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["HybridFCL"]


def HybridFCL():
    class HybridFCLModule(nn.Module):
        """Classical two‑layer network with softmax output."""

        def __init__(self, n_features: int = 2) -> None:
            super().__init__()
            # Hidden layer: n_features → 4
            self.linear1 = nn.Linear(n_features, 4)
            # Output layer: 4 → 2
            self.linear2 = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            """Forward pass: returns softmax probabilities."""
            hidden = torch.tanh(self.linear1(x))
            logits = self.linear2(hidden)
            probs = F.softmax(logits, dim=-1)
            return probs

        def run(self, thetas: list[float]) -> np.ndarray:
            """
            Accepts a list of parameters, feeds them through the network
            and returns a tuple ``(probs, expectation)``.
            """
            values = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
            probs = self.forward(values)
            # Expectation: mean of tanh‑activated logits
            expectation = torch.tanh(self.linear2(torch.tanh(self.linear1(values)))).mean(dim=0)
            return probs.detach().numpy(), expectation.detach().numpy()

    return HybridFCLModule()
