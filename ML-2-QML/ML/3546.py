from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

class SamplerQNN(nn.Module):
    """Classical sampler mimicking the Qiskit SamplerQNN architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud detection model that layers photonic-inspired
    transformations and a quantum-inspired sampler network.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Sequence[FraudLayerParameters]) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules += [_layer_from_params(layer, clip=True) for layer in layers]
        modules.append(nn.Linear(2, 1))
        self.feature_extractor = nn.Sequential(*modules)
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        probs = self.sampler(features)
        return probs

    def quantum_parameters(self) -> dict[str, torch.Tensor]:
        """
        Exposes the learned parameters that will feed into the quantum circuit.
        """
        params = {}
        linear = self.feature_extractor[-1]
        params["bias"] = linear.bias.clone()
        for name, module in self.sampler.named_modules():
            if isinstance(module, nn.Linear):
                params[f"{name}.weight"] = module.weight.clone()
                params[f"{name}.bias"] = module.bias.clone()
        return params
