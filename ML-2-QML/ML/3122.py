"""Hybrid fraud detection architecture combining photonic and quantum layers.

This module merges the ideas from the two reference pairs:
* The photonic‑style layer construction from the first pair is used as a
  feature extractor, but its parameters are now trainable and grouped into
  a single `FraudDetectionHybridNet` class.
* A Qiskit‑based one‑qubit variational circuit, inspired by the second pair,
  is attached as a differentiable expectation head.
* Back‑propagation is possible because the quantum expectation is wrapped in
  a custom autograd function that implements a finite‑difference gradient.

The network can be trained end‑to‑end with any PyTorch optimiser and
produces a two‑class probability vector [p, 1‑p].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble


@dataclass
class FraudLayerParameters:
    """Container for a photonic layer's trainable parameters."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    """Build a single photonic‑style linear block."""
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            y = y * self.scale + self.shift
            return y

    return Layer()


class QuantumCircuit:
    """Simple one‑qubit variational circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = self.circuit.add_parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        expectations = []
        for p in params:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: float(p)}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            prob0 = counts.get("0", 0) / self.shots
            prob1 = counts.get("1", 0) / self.shots
            expectations.append(prob0 - prob1)
        return np.array(expectations)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        eps = 1e-3
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run(np.array([val + eps]))[0]
            left = ctx.circuit.run(np.array([val - eps]))[0]
            grad = (right - left) / (2 * eps)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(
            grad_inputs, dtype=inputs.dtype, device=inputs.device
        )
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(-1), self.quantum_circuit, self.shift)


class FraudDetectionHybridNet(nn.Module):
    """Full hybrid fraud‑detection network."""

    def __init__(self, n_layers: int = 3, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        # Classical feature extractor
        self.classical_layers = nn.ModuleList()
        input_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.5,
            phases=(0.1, 0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.1, 0.1),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.classical_layers.append(_layer_from_params(input_params, clip=False))
        for _ in range(n_layers - 1):
            layer_params = FraudLayerParameters(
                bs_theta=np.random.rand(),
                bs_phi=np.random.rand(),
                phases=(np.random.rand(), np.random.rand()),
                squeeze_r=(np.random.rand(), np.random.rand()),
                squeeze_phi=(np.random.rand(), np.random.rand()),
                displacement_r=(np.random.rand(), np.random.rand()),
                displacement_phi=(np.random.rand(), np.random.rand()),
                kerr=(np.random.rand(), np.random.rand()),
            )
            self.classical_layers.append(_layer_from_params(layer_params, clip=True))
        self.final_linear = nn.Linear(2, 1)

        # Quantum head
        backend = Aer.get_backend("aer_simulator")
        self.quantum_head = Hybrid(1, backend, shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.classical_layers:
            x = layer(x)
        x = self.final_linear(x)
        x = x.squeeze(-1)
        q_out = self.quantum_head(x)
        prob = torch.sigmoid(q_out)
        return torch.stack([prob, 1 - prob], dim=-1)


__all__ = [
    "FraudLayerParameters",
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "FraudDetectionHybridNet",
]
