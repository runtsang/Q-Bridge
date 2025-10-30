"""Quantum‑enhanced classifier that mirrors the classical backbone and adds a
differentiable quantum head.  The implementation is self‑contained and
requires only qiskit and torch.

The module retains the classical components (CNN, fraud‑detection layers,
RBF kernel) from the seed while replacing the expectation head with a
parameterised quantum circuit that is differentiable via a custom
autograd function.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
# Classical components (re‑implemented for compatibility)
# --------------------------------------------------------------------------- #
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    out_features: int = 84,
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, out_features))
    return nn.Sequential(*modules)

class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel used as an auxiliary feature."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Quantum components
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parameterized one‑qubit circuit that returns the expectation of Z."""
    def __init__(self, backend, shots: int = 100):
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = {k: v / self.shots for k, v in counts.items()}
            return probs.get("0", 0) - probs.get("1", 0)
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(thetas)
        result = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

class QuantumHybridHead(nn.Module):
    """Hybrid head that maps classical logits through a quantum circuit."""
    def __init__(self, backend, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(logits, self.circuit, self.shift)

class HybridQuantumClassifier(nn.Module):
    """Quantum‑enhanced classifier mirroring the classical backbone."""
    def __init__(self, quantum_backend, shots: int = 100) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        self.fc_reduce = nn.Linear(60, 2)
        input_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud = build_fraud_detection_program(input_params, [], out_features=84)
        self.kernel = RBFKernel(gamma=0.5)
        self.kernel_vector = nn.Parameter(torch.randn(84))
        self.fc = nn.Linear(85, 1)
        self.quantum_head = QuantumHybridHead(quantum_backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc_reduce(x)
        x = self.fraud(x)
        k = self.kernel(x, self.kernel_vector)
        logits = self.fc(torch.cat([x, k], dim=-1))
        q_logits = self.quantum_head(logits)
        probs = torch.sigmoid(q_logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "QuantumHybridHead",
    "HybridQuantumClassifier",
]
