"""Classical sampler network mirroring the QNN helper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def SamplerQNN():
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


__all__ = ["SamplerQNN"]
