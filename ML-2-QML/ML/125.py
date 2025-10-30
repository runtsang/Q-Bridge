"""Enhanced classical feed‑forward regressor for regression tasks.

The network architecture is extended with batch‑normalization and dropout layers
to improve generalisation.  The `EstimatorQNN` function returns an instance of
the model ready for training with any PyTorch optimiser.

The class is intentionally lightweight so that it can be swapped into the
quantum module for hybrid training experiments.
"""

from __future__ import annotations

import torch
from torch import nn


def EstimatorQNN() -> nn.Module:
    """Return a robust feed‑forward regression network.

    Architecture:
        - Input layer: 2 → 16
        - BatchNorm1d
        - ReLU
        - Linear 16 → 32
        - Dropout(0.2)
        - Linear 32 → 16
        - BatchNorm1d
        - ReLU
        - Linear 16 → 1

    Returns
    -------
    nn.Module
        Fully‑connected regression model.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
