"""Hybrid sampler QNN combining SamplerQNN and FraudDetection concepts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class _FraudLayer(nn.Module):
    def __init__(self, params: dict, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [
                [params["bs_theta"], params["bs_phi"]],
                [params["squeeze_r"][0], params["squeeze_r"][1]],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(params["phases"], dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params["displacement_r"], dtype=torch.float32)
        self.shift = torch.tensor(params["displacement_phi"], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

class HybridSamplerQNN(nn.Module):
    def __init__(self, input_params: dict, layers: list[dict]) -> None:
        super().__init__()
        modules = [_FraudLayer(input_params, clip=False)]
        modules.extend(_FraudLayer(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

def SamplerQNN() -> HybridSamplerQNN:
    """Factory returning a HybridSamplerQNN instance."""
    input_params = {
        "bs_theta": 0.5,
        "bs_phi": 0.3,
        "phases": (0.1, -0.1),
        "squeeze_r": (0.2, 0.2),
        "squeeze_phi": (0.0, 0.0),
        "displacement_r": (1.0, 1.0),
        "displacement_phi": (0.0, 0.0),
        "kerr": (0.0, 0.0),
    }
    layers = [input_params]  # placeholder; real layers should be provided
    return HybridSamplerQNN(input_params, layers)

__all__ = ["HybridSamplerQNN", "SamplerQNN"]
