"""Combined classical implementation of a hybrid fully‑connected layer.

The model mirrors the quantum‑inspired structure from the original
`FCL.py`, `FraudDetection.py` and `QCNN.py` seeds.  A fraud‑detection
layer is followed by a classical QCNN stack and finished with a
fully‑connected head.  The final activation is a differentiable
sigmoid that mimics the quantum expectation value used in the
quantum counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Fraud‑detection block – same parameter layout as the photonic seed
# ------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Classical QCNN – emulation of the quantum convolution + pooling
# ------------------------------------------------------------------
class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ------------------------------------------------------------------
# Hybrid head – differentiable sigmoid that emulates a quantum
# expectation value
# ------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces a quantum expectation layer."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# ------------------------------------------------------------------
# The combined classical model
# ------------------------------------------------------------------
class FCLHybrid(nn.Module):
    """Classical implementation of a hybrid fully‑connected layer.

    The network consists of:
      * a fraud‑detection block (parameterised 2‑to‑2 linear layers)
      * a classical QCNN stack that mimics the quantum convolution‑pool
        pattern
      * a final fully‑connected head with a quantum‑style sigmoid
    """
    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        *,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.fraud = build_fraud_detection_program(fraud_input, fraud_layers)
        self.qcnn = QCNNModel()
        self.hybrid = Hybrid(1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fraud(inputs)
        x = self.qcnn(x)
        x = self.hybrid(x)
        return x

def FCLHybridFactory(
    fraud_input: FraudLayerParameters,
    fraud_layers: Iterable[FraudLayerParameters],
    *,
    shift: float = 0.0,
) -> FCLHybrid:
    """Convenience factory returning a fully‑wired instance."""
    return FCLHybrid(fraud_input, fraud_layers, shift=shift)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "QCNNModel", "HybridFunction", "Hybrid", "FCLHybrid", "FCLHybridFactory"]
