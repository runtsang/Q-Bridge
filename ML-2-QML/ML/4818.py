"""Hybrid classical kernel combining RBF with fraud‑detection style linear
transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer used for data
    pre‑processing before kernel evaluation."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudLinear(nn.Module):
    """Linear transform mirroring a single fraud‑detection layer."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(inputs)


class HybridKernel(nn.Module):
    """RBF kernel with optional fraud‑detection style preprocessing."""
    def __init__(self, gamma: float = 1.0,
                 preproc_params: FraudLayerParameters | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.preproc = FraudLinear(preproc_params, clip=False) if preproc_params else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.preproc(x).view(-1, 2)
        y = self.preproc(y).view(-1, 2)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  preproc_params: FraudLayerParameters | None = None) -> np.ndarray:
    kernel = HybridKernel(gamma, preproc_params)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["FraudLayerParameters", "HybridKernel", "kernel_matrix"]
