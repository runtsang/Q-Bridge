from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen128(nn.Module):
    """Classical deep sampler producing a 128‑class probability distribution.

    Two input features are projected into a high‑dimensional hidden space
    before being collapsed to 128 logits. Softmax yields a valid
    probability distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


def SamplerQNN() -> SamplerQNNGen128:
    """Convenience factory that matches the original API."""
    return SamplerQNNGen128()


__all__ = ["SamplerQNNGen128", "SamplerQNN"]
