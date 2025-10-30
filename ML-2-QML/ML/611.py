"""Enhanced classical sampler with dropout, batch‑norm and sampling API."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["SamplerQNN"]


def SamplerQNN() -> nn.Module:
    """
    Returns a torch.nn.Module that implements a 2‑to‑2 sampler with
    dropout, batch‑normalization and a convenient ``sample`` method.
    The architecture is deliberately deeper than the seed to
    demonstrate how classical models can be scaled while keeping
    the public API identical.
    """
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Two hidden layers with batch‑norm and dropout
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(8, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(8, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            logits = self.net(inputs)
            return F.softmax(logits, dim=-1)

        def sample(
            self, inputs: torch.Tensor, n_samples: int = 1, device: torch.device | str = "cpu"
        ) -> torch.Tensor:
            """
            Draw samples from the categorical distribution defined by the
            network output.  This is useful for Monte‑Carlo style experiments
            and for feeding the sampler into downstream pipelines.
            """
            self.eval()
            probs = self.forward(inputs.to(device))
            # Expand to batch dimension if necessary
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            samples = torch.multinomial(probs, n_samples, replacement=True)
            return samples

    return SamplerModule()
