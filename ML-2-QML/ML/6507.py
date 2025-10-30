from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from typing import Iterable

# ----- Classical fraud‑detection layer (from FraudDetection.py) -----

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

# ----- Classical CNN backbone (from ClassicalQuantumBinaryClassification.py) -----

class FraudCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x

# ----- Quantum circuit (parameterised two‑qubit) -----

class QuantumCircuit:
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, angles: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: angle} for angle in angles],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(c) for c in result])
        return np.array([self._expectation(result)])

    def _expectation(self, counts: dict) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.sum(states * probs)

# ----- Autograd bridge between PyTorch and the quantum circuit -----

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.tolist()
        exp_vals = circuit.run(np.array(angles))
        out = torch.tensor(exp_vals, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.tolist()
        grad = []
        for a in angles:
            e_plus = ctx.circuit.run(np.array([a + shift]))
            e_minus = ctx.circuit.run(np.array([a - shift]))
            grad.append(e_plus - e_minus)
        grad = torch.tensor(grad, dtype=torch.float32)
        return grad * grad_output, None, None

# ----- Hybrid fraud‑detection model -----

class FraudDetectionHybrid(nn.Module):
    def __init__(self, cnn: nn.Module | None = None, quantum_circuit: QuantumCircuit | None = None) -> None:
        super().__init__()
        self.cnn = cnn if cnn is not None else FraudCNN()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum = quantum_circuit if quantum_circuit is not None else QuantumCircuit(2, backend, shots=100)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        probs = HybridFunction.apply(features, self.quantum, self.shift)
        probs = torch.sigmoid(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
