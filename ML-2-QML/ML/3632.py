from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, Tensor

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single linear‑Tanh‑scale layer from the given parameters."""
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
            self.dropout = nn.Dropout(p=0.1)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = self.dropout(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybridML(nn.Module):
    """Hybrid classical model that optionally uses a probabilistic sampler."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layers)
        self.use_sampler = use_sampler
        if use_sampler:
            from.SamplerQNN import SamplerQNN

            self.sampler = SamplerQNN()

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        if self.use_sampler:
            # The SamplerQNN expects a 2‑dim input; logits are 1‑dim so we pad.
            logits = logits.expand(-1, 2)
            probs = self.sampler(logits)
            return probs
        return logits


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybridML"]
