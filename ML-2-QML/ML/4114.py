import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Classical softmax sampler (from SamplerQNN.py)
# ----------------------------------------------------------------------
class ClassicalSampler(nn.Module):
    """Lightweight 2‑input softmax network used as a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# ----------------------------------------------------------------------
# Photonic fraud‑detecting layer (from FraudDetection.py)
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool = True) -> nn.Module:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Placeholder hybrid function for classical‑only mode
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable layer that mimics a quantum expectation using a sigmoid."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        y = torch.sigmoid(x + shift)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (y,) = ctx.saved_tensors
        return grad_output * y * (1 - y), None

class HybridLayer(nn.Module):
    """Layer that forwards activations through the placeholder quantum head."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.shift)

# ----------------------------------------------------------------------
# Unified hybrid sampler
# ----------------------------------------------------------------------
class UnifiedHybridSampler(nn.Module):
    """End‑to‑end sampler combining classical sampling, fraud‑detection regularisation,
    and a quantum‑enhanced (placeholder) decision head."""
    def __init__(self, n_fraud_layers: int = 3, shift: float = 0.0) -> None:
        super().__init__()
        self.sampler = ClassicalSampler()
        # Build fraud detection program with dummy parameters
        dummy_params = FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.2,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud = build_fraud_detection_program(
            dummy_params,
            [dummy_params for _ in range(n_fraud_layers)],
        )
        self.hybrid = HybridLayer(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sampler(x)
        out = self.fraud(out)
        out = self.hybrid(out.squeeze(-1))
        return out

__all__ = ["UnifiedHybridSampler"]
