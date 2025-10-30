"""
HybridQCNet – quantum‑enhanced binary classifier.
It replaces the classical final head with a parameterised two‑qubit
quantum circuit that outputs the expectation of the Z‑operator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import qiskit
from qiskit import assemble, transpile

__all__ = ["FraudLayerParameters", "FraudLayer", "HybridQCNet"]


@dataclass
class FraudLayerParameters:
    """Parameters for the photonic‑style fully‑connected layer."""
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


class FraudLayer(nn.Module):
    """
    Classical photonic‑style layer used as a feature encoder before the quantum head.
    Mirrors the structure of the photonic circuit but remains fully classical.
    """

    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
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

        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.linear(inputs))
        return x * self.scale + self.shift


class QuantumCircuit:
    """
    Two‑qubit parametric circuit used as the quantum expectation head.
    The circuit is H → RY(θ) → measure all; expectation of Z is returned.
    """

    def __init__(self, backend, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        qubits = list(range(2))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.ry(self.theta, qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])  # binary to int
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards input angles to the quantum circuit
    and applies the parameter‑shift rule for gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        exp_val = circuit.run(angles)
        result = torch.tensor(exp_val, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grad_inputs.append(right - left)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """
    Quantum expectation layer that replaces the classical final head.
    """

    def __init__(self, backend, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class HybridQCNet(nn.Module):
    """
    Convolutional network followed by a fraud‑style layer and a quantum head.
    Mirrors the classical version but uses a quantum expectation head for the final output.
    """

    def __init__(self,
                 fraud_params: FraudLayerParameters,
                 backend,
                 shots: int = 1024,
                 shift: float = np.pi / 2,
                 fraud_clip: bool = True) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected blocks
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 outputs for fraud layer

        # Fraud‑style layer
        self.fraud = FraudLayer(fraud_params, clip=fraud_clip)

        # Quantum head
        self.quantum = Hybrid(backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.fraud(x)
        # Expectation value from quantum circuit
        quantum_out = self.quantum(x).squeeze(-1)
        probs = torch.sigmoid(quantum_out)
        return torch.cat((probs, 1 - probs), dim=-1)
