"""Hybrid quantum‑classical binary classifier with photonic‑inspired preprocessing (quantum implementation)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from dataclasses import dataclass
from typing import Iterable, Tuple

# ------------------------------------------------------------
# Photonic‑inspired preprocessing layers (identical to ML side)
# ------------------------------------------------------------
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------
# Parameterised quantum circuit
# ------------------------------------------------------------
class QuantumCircuit:
    """Two‑qubit circuit with H, Ry, and measurement of all qubits."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# ------------------------------------------------------------
# Autograd integration using parameter‑shift rule
# ------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist()).item()
        result = torch.tensor([expectation_z], dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for idx, value in enumerate(inputs.numpy()):
            right = ctx.circuit.run([value + shift[idx]]).item()
            left = ctx.circuit.run([value - shift[idx]]).item()
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None

# ------------------------------------------------------------
# Hybrid layer that injects quantum expectation into the network
# ------------------------------------------------------------
class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# ------------------------------------------------------------
# Main hybrid network
# ------------------------------------------------------------
class QCNet(nn.Module):
    def __init__(self,
                 fraud_input: FraudLayerParameters,
                 fraud_layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        # Classical fraud‑detection preprocessing
        self.fraud_prog = build_fraud_detection_program(fraud_input, fraud_layers)

        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi/2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Apply fraud preprocessing on first two channels
        fraud_input = inputs[:, :2, :, :]
        fraud_features = self.fraud_prog(fraud_input.view(fraud_input.size(0), -1))

        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Concatenate classical and conv features
        x = torch.cat([x, fraud_features], dim=1)

        # Fully connected forward
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid quantum head
        x = self.hybrid(x).transpose(0, 1)

        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
