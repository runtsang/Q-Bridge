"""Hybrid classical‑quantum model with a quantum expectation head.

The architecture mirrors the classical branch but replaces the final
dense head with a parameterised Qiskit circuit.  The model remains
fully differentiable through a custom autograd function that
evaluates the circuit on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.circuit.random import random_circuit
from dataclasses import dataclass
from typing import Iterable


# -------------------------------------------------------------
# 1. Quantum circuit wrapper
# -------------------------------------------------------------
class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on a Qiskit backend."""

    def __init__(self, n_qubits: int, backend, shots: int, threshold: float) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of input angles."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)

        def expectation(counts: dict[str, int]) -> float:
            total = 0
            for bitstring, cnt in counts.items():
                ones = sum(int(b) for b in bitstring)
                total += ones * cnt
            return total / (self.shots * self.n_qubits)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


# -------------------------------------------------------------
# 2. Hybrid sigmoid head (quantum expectation)
# -------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards through a Qiskit circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit on CPU
        expect = ctx.circuit.run(inputs.numpy())
        out = torch.tensor(expect, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.numpy(), shift):
            right = ctx.circuit.run(np.array([val + s]))
            left = ctx.circuit.run(np.array([val - s]))
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Layer that runs a quantum circuit and applies a sigmoid."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots, threshold=0.5)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)


# -------------------------------------------------------------
# 3. Fraud‑Detection inspired dense head (same as classical)
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 4. Main hybrid model
# -------------------------------------------------------------
class HybridQuantumNAT(nn.Module):
    """Convolutional backbone + fraud‑like dense stack + quantum expectation head."""

    def __init__(self, n_classes: int = 1, shift: float = np.pi / 2) -> None:
        super().__init__()
        # Convolutional backbone (same as QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce to 2‑dimensional representation
        self.fc_reduce = nn.Linear(16 * 7 * 7, 2)

        # Fraud‑detection inspired dense stack
        input_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.4, 0.4),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=0.6,
                bs_phi=0.4,
                phases=(0.2, -0.2),
                squeeze_r=(0.3, 0.3),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.5, 0.5),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            for _ in range(2)
        ]
        self.fraud_head = build_fraud_detection_program(input_params, layers)

        # Quantum head
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=200, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        reduced = self.fc_reduce(flattened)
        fraud_output = self.fraud_head(reduced)  # shape (bsz, 1)
        # The fraud head already outputs a scalar; we feed it to the quantum head
        quantum_expect = self.hybrid(fraud_output)
        return quantum_expect


__all__ = ["HybridQuantumNAT", "HybridFunction", "Hybrid", "FraudLayerParameters", "build_fraud_detection_program"]
