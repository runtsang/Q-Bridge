"""FraudDetectionDual – classical residual‑dense network with quantum‑derived embeddings.

The module implements a residual‑dense block that uses the parameters from the
seed model.  It can be combined with the quantum embedding defined in the QML
module to form a hybrid classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["FraudLayerParameters", "ResidualDenseBlock", "build_fraud_detection_dual"]

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    attention_mask: float = 1.0
    scale_gate: float = 1.0

class ResidualDenseBlock(nn.Module):
    """Residual‑dense block that stacks several parameterised linear layers.

    Each layer is constructed from the original parameters and a skip connection
    that is gated by ``attention_mask``.  The output of the block is further
    scaled by ``scale_gate``.
    """
    def __init__(self, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.attention_masks = []
        self.scale_gates = []
        for p in layers:
            weight = torch.tensor(
                [[p.bs_theta, p.bs_phi],
                 [p.squeeze_r[0], p.squeeze_r[1]]],
                dtype=torch.float32)
            bias = torch.tensor(p.phases, dtype=torch.float32)
            linear = nn.Linear(2, 2, bias=True)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            self.layers.append(nn.Sequential(
                linear,
                nn.Tanh()
            ))
            self.attention_masks.append(p.attention_mask)
            self.scale_gates.append(p.scale_gate)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.float32)
        self.scale_gates = torch.tensor(self.scale_gates, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, layer in enumerate(self.layers):
            residual = out
            out = layer(out)
            out = out * self.attention_masks[idx] + residual
        out = out * self.scale_gates.sum()
        return out

def build_fraud_detection_dual(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Create a sequential model that starts with a single input layer and
    then applies the ResidualDenseBlock, followed by a final linear classifier.
    """
    # Input layer
    input_weight = torch.tensor(
        [[input_params.bs_theta, input_params.bs_phi],
         [input_params.squeeze_r[0], input_params.squeeze_r[1]]],
        dtype=torch.float32)
    input_bias = torch.tensor(input_params.phases, dtype=torch.float32)
    input_linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        input_linear.weight.copy_(input_weight)
        input_linear.bias.copy_(input_bias)

    seq = nn.Sequential(
        input_linear,
        nn.Tanh(),
        ResidualDenseBlock(layers),
        nn.Linear(2, 1)
    )
    return seq
