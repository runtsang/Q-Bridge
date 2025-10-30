"""Hybrid classical estimator that mimics fraudulent detection architecture.

This model extends the original EstimatorQNN by allowing a variable number of
custom layers, each parameterised by a FraudLayerParameters dataclass.
The first layer is unregularised, subsequent layers are clipped to keep
weights within a safe range.  The network ends with a single neuron that
produces a scalar output suitable for regression or binary classification.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class EstimatorQNN(nn.Module):
    """A feed‑forward network with fraud‑layer inspired blocks."""
    def __init__(self, layers: List[FraudLayerParameters]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx, params in enumerate(layers):
            clip = idx > 0
            block = self._build_block(params, clip=clip)
            self.blocks.append(block)
        self.out = nn.Linear(2, 1)

    def _build_block(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                y = self.activation(self.linear(x))
                return y * self.scale + self.shift

        return Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)

__all__ = ["EstimatorQNN", "FraudLayerParameters"]
