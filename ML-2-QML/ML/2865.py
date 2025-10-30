"""Hybrid classical neural network mirroring a quantum sampler‑estimator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerEstimatorQNN(nn.Module):
    """A hybrid network that combines a sampler and an estimator.

    The sampler produces a probability distribution over two classes
    using a small feed‑forward network.  The estimator predicts a
    scalar value based on the same input.  Both sub‑networks are
    trained jointly, allowing the classical model to emulate a
    quantum hybrid architecture while remaining fully differentiable
    in PyTorch.
    """
    def __init__(self) -> None:
        super().__init__()
        # Sampler head: 2 → 6 → 2
        self.sampler = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 2),
        )
        # Estimator head: 2 → 8 → 4 → 1
        self.estimator = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (probabilities, estimate)."""
        probs = F.softmax(self.sampler(inputs), dim=-1)
        estimate = self.estimator(inputs)
        return probs, estimate

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns only the estimator output."""
        return self.estimator(inputs)

__all__ = ["HybridSamplerEstimatorQNN"]
