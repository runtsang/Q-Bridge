"""Classical hybrid layer combining linear, feed‑forward, and shift components.

This module implements `UnifiedHybridLayer`, a pure‑Python PyTorch
module that mimics the behaviour of the original FCL example while
incorporating the EstimatorQNN regressor and a learnable sigmoid shift.
The `run` method accepts a list of angles and returns a mean expectation
value in a NumPy array, matching the interface of the quantum seed.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable

__all__ = ["UnifiedHybridLayer"]

class UnifiedHybridLayer(nn.Module):
    """Classical hybrid layer.

    Attributes
    ----------
    linear : nn.Linear
        Linear head that maps the input feature(s) to a single output.
    regressor : nn.Sequential
        Feed‑forward network mirroring the EstimatorQNN architecture.
    shift : float
        Learnable shift added before the sigmoid activation in ``forward``.
    """

    def __init__(self, n_features: int = 1, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.shift = shift
        # Feed‑forward regressor
        self.regressor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Forward pass that accepts a list of thetas and returns a mean expectation."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Combines the linear head with a sigmoid activation and a learnable shift."""
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)
