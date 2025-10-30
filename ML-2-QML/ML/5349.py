import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Callable, List, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs
    return Layer()

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_callable: Callable[[torch.Tensor], torch.Tensor], shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_callable = quantum_callable
        expectation = ctx.quantum_callable(inputs)
        ctx.save_for_backward(inputs, expectation)
        return expectation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        inputs, expectation = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.squeeze().tolist():
            right = ctx.quantum_callable(torch.tensor([val + shift]))
            left = ctx.quantum_callable(torch.tensor([val - shift]))
            grad_inputs.append((right - left).item())
        grad = torch.tensor(grad_inputs, dtype=torch.float32).unsqueeze(0)
        return grad * grad_output, None, None

class HybridLayer(nn.Module):
    """Wraps a quantum callable as a differentiable layer."""
    def __init__(self, quantum_callable: Callable[[torch.Tensor], torch.Tensor], shift: float = 0.0):
        super().__init__()
        self.quantum_callable = quantum_callable
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_callable, self.shift)

class FraudDetectionHybrid(nn.Module):
    """Classical backbone with a quantum expectation head for fraud detection."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: List[FraudLayerParameters],
        quantum_callable: Callable[[torch.Tensor], torch.Tensor],
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.backbone = nn.Sequential(*modules)
        self.hybrid = HybridLayer(quantum_callable, shift)
        self.regressor = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        prob = self.hybrid(x)
        prob = torch.sigmoid(prob)
        probs = torch.cat([prob, 1 - prob], dim=-1)
        regression = self.regressor(prob)
        return probs, regression

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Utility to construct the classical backbone without the quantum head."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
