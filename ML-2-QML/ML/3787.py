"""Hybrid fraud‑detection pipeline with quantum‑inspired convolution.

This module combines the photonic‑style fully‑connected layers from
FraudDetection.py with the classical convolution filter from Conv.py.
The resulting API can build a pure classical model or a hybrid model
that inserts a quantum‑inspired convolution step before the linear stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

# ----------------------------------------------------------------------
# 1. Parameter container (identical to the photonic version)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2. Utility clipping
# ----------------------------------------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# ----------------------------------------------------------------------
# 3. Layer factory – mirrors the photonic linear block
# ----------------------------------------------------------------------
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# ----------------------------------------------------------------------
# 4. Pure classical stack (photonic‑style)
# ----------------------------------------------------------------------
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# 5. Classical convolution filter – identical to Conv.py
# ----------------------------------------------------------------------
def Conv() -> nn.Module:
    """Return a PyTorch module that emulates a quantum convolution filter."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            # `data` shape: (batch, 1, H, W)
            logits = self.conv(data)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean(dim=[1, 2, 3])

    return ConvFilter()

# ----------------------------------------------------------------------
# 6. Hybrid builder – inserts the convolution before the linear stack
# ----------------------------------------------------------------------
def build_fraud_detection_program_with_conv(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    conv_kernel: int = 2,
    threshold: float = 0.0,
) -> nn.Sequential:
    """Build a stack that starts with a quantum‑inspired convolution."""
    conv = Conv()
    # The convolution outputs a scalar; map it to a 2‑D vector for the stack.
    modules: list[nn.Module] = [
        conv,
        nn.Linear(1, 2),
        _layer_from_params(input_params, clip=False),
    ]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_fraud_detection_program_with_conv",
    "Conv",
]
