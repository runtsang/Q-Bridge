from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List, Optional

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

def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))

def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
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
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out
    return Layer()

def build_fraud_detection_network(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules += [_layer_from_params(l, clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FCLGen287(nn.Module):
    """Hybrid fully‑connected layer that combines a fraud‑detection inspired
    classical network with optional quantum input features.

    The network is built from a list of :class:`FraudLayerParameters`.  The
    first layer is unconstrained; subsequent layers are clipped to keep the
    parameters within a reasonable range.  An optional quantum feature
    vector (e.g. expectation values from a variational circuit) can be
    concatenated to the input and fed through the same network.

    The design mirrors the two reference seeds: the classical part follows
    the `FraudDetection` construction, while the quantum contribution is
    compatible with the `FCL` variational routine.
    """
    def __init__(
        self,
        params: Iterable[FraudLayerParameters],
        *,
        quantum_dim: int = 0,
    ) -> None:
        super().__init__()
        self.quantum_dim = quantum_dim
        self.net = build_fraud_detection_network(params[0], params[1:])
        if quantum_dim > 0:
            self.prepend = nn.Linear(quantum_dim, 2)
        else:
            self.prepend = None

    def forward(self, x: torch.Tensor, q_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prepend and q_feat is not None:
            x = torch.cat([self.prepend(q_feat), x], dim=-1)
        return self.net(x)

__all__ = ["FraudLayerParameters", "build_fraud_detection_network", "FCLGen287"]
