"""Deep classical regression model for the EstimatorQNN family.

This module replaces the original 2‑layer toy network with a
regularised feed‑forward network that includes batch‑normalisation,
dropout and a configurable hidden‑layer sequence.  The class
`EstimatorQNN` is intentionally lightweight so that it can be
instantiated in the same way as the quantum counterpart, allowing
parallel experimentation across classical and quantum back‑ends."""
from __future__ import annotations

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    A deeper, regularised feed‑forward network that mirrors the
    functionality of the original EstimatorQNN but with improved
    expressivity and training stability.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: list[int] | tuple[int,...] = (32, 16),
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                ]
            )
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar regression output."""
        return self.net(x)


__all__ = ["EstimatorQNN"]
