"""Hybrid fraud detection model combining classical quantum‑inspired convolution
and photonic‑inspired linear layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric bound."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a photonic‑inspired linear layer."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class ClassicalQuanvolutionFilter(nn.Module):
    """Classical approximation of a quantum convolutional filter using a random unitary."""
    def __init__(self) -> None:
        super().__init__()
        # Random orthogonal unitary (4x4)
        q, _ = torch.linalg.qr(torch.randn(4, 4))
        self.register_buffer("unitary", q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        returns: (batch, 4*14*14)
        """
        bsz, _, h, w = x.shape
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.permute(0, 2, 3, 4, 5).contiguous()  # (batch, 14, 14, 2, 2)
        patches = patches.view(bsz, 14 * 14, 4)  # (batch, 196, 4)
        # Apply unitary to each patch
        out = torch.einsum("bnp,pr->bnr", patches, self.unitary.T)
        out = out.view(bsz, -1)  # (batch, 196*4)
        return out


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model that first applies a classical quantum‑inspired
    convolutional filter and then a stack of photonic‑inspired linear layers.
    """
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.quantum_conv = ClassicalQuanvolutionFilter()
        self.reduction = nn.Linear(4 * 14 * 14, 2)
        self.photonic = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *[_layer_from_params(l, clip=True) for l in layers],
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_conv(x)
        x = self.reduction(x)
        x = self.photonic(x)
        return x


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
