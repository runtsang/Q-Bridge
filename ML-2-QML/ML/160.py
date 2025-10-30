"""Enhanced classical sampler network with residual, dropout, and batch‑norm layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def SamplerQNN() -> nn.Module:
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Residual block 1
            self.block1 = nn.Sequential(
                nn.Linear(2, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            # Residual block 2
            self.block2 = nn.Sequential(
                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.out = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.block1(inputs)
            x = self.block2(x)
            logits = self.out(x)
            return F.softmax(logits, dim=-1)

    return SamplerModule()


__all__ = ["SamplerQNN"]
