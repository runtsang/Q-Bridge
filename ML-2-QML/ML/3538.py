"""Hybrid classical convolution + fraud‑detection network.

This module preserves the public API of the original Conv.py but adds a
fully‑connected stack inspired by the FraudDetection example.  The
`Conv()` factory now accepts optional `layers_params`, a sequence of
`FraudLayerParameters` that describe each fully‑connected layer.  The
convolution step produces a 2‑element feature vector (mean and std) that
feeds into the FC stack, yielding a scalar output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for one fraud‑detection style fully‑connected layer."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class ConvFilter(nn.Module):
    """Drop‑in replacement for the original quanvolution filter.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the square convolution kernel.
    threshold : float, default=0.0
        Threshold for the sigmoid activation.
    layers_params : Iterable[FraudLayerParameters] | None, default=None
        Sequence of parameters that build a fraud‑detection style FC stack.
        If ``None`` the output of the convolution is returned directly.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        layers_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Build fraud‑detection style stack if requested.
        if layers_params is None:
            self.fc_stack = nn.Identity()
        else:
            modules = [_layer_from_params(l, clip=False)]
            modules += [_layer_from_params(l, clip=True) for l in layers_params]
            modules.append(nn.Linear(2, 1))
            self.fc_stack = nn.Sequential(*modules)

    def run(self, data) -> float:
        """Apply convolution, compute mean/std and pass through FC stack."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)

        # Build 2‑D feature vector: mean and std of activations.
        feat = torch.tensor(
            [activations.mean().item(), activations.std().item()],
            dtype=torch.float32,
        )
        out = self.fc_stack(feat)
        return out.item()


def Conv(
    kernel_size: int = 2,
    threshold: float = 0.0,
    layers_params: Iterable[FraudLayerParameters] | None = None,
) -> ConvFilter:
    """Factory that returns a drop‑in compatible convolution filter."""
    return ConvFilter(kernel_size, threshold, layers_params)


__all__ = ["Conv", "FraudLayerParameters"]
