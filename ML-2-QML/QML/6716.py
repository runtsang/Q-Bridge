from dataclasses import dataclass
import pennylane as qml
import torch
from torch import nn
import numpy as np

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

def _construct_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_construct_layer(input_params, clip=False)]
    modules.extend(_construct_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuantumKernel(nn.Module):
    """Quantum kernel based on a variational Ry circuit and CNOT entanglement."""
    def __init__(self) -> None:
        super().__init__()
        self.dev = qml.device("default.qubit", wires=4)
        self._circuit = qml.QNode(self._build_circuit, self.dev, interface="torch", diff_method="adjoint")

    def _build_circuit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        for i, val in enumerate(y):
            qml.RY(-val, wires=i)
        return qml.state()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state_x = self._circuit(x, torch.zeros_like(x))
        state_y = self._circuit(torch.zeros_like(x), y)
        return torch.abs(torch.dot(state_x, state_y.conj()))

class FraudDetectionHybrid(nn.Module):
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        support_vectors: torch.Tensor,
        kernel: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.classical_net = build_fraud_detection_program(input_params, layers)
        self.register_buffer("support_vectors", support_vectors)
        self.kernel = kernel or QuantumKernel()
        self.classifier = nn.Linear(2 + support_vectors.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_feat = self.classical_net(x).flatten(-1)
        kernel_feat = self.kernel(x, self.support_vectors)
        combined = torch.cat([class_feat, kernel_feat], dim=-1)
        return self.classifier(combined)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "QuantumKernel", "FraudDetectionHybrid"]
