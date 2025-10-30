"""Hybrid classical-quantum binary classifier with photonic-inspired classical layers and a quantum expectation head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

import qiskit
from qiskit import assemble, transpile

@dataclass
class ClassicalFraudLayerParameters:
    """Parameters for a photonic‑inspired linear layer with clipping, scale and shift."""
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

def _layer_from_params(params: ClassicalFraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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

def build_classical_fraud_program(
    input_params: ClassicalFraudLayerParameters,
    layers: Iterable[ClassicalFraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit

        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QCNetHybrid(nn.Module):
    """Convolutional backbone + photonic-inspired classical layers + quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 outputs feed into classical seq

        # Classical photonic-inspired sequence
        default_input = ClassicalFraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3, phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        default_layers = [
            ClassicalFraudLayerParameters(
                bs_theta=0.4, bs_phi=0.2, phases=(0.05, -0.05),
                squeeze_r=(0.1, 0.1), squeeze_phi=(0.0, 0.0),
                displacement_r=(0.8, 0.8), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0)
            )
        ]
        self.classical_seq = build_classical_fraud_program(default_input, default_layers)

        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=1, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape (batch, 2)
        x = self.classical_seq(x)  # shape (batch, 1)
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = [
    "ClassicalFraudLayerParameters",
    "build_classical_fraud_program",
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "QCNetHybrid",
]
