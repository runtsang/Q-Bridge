"""Hybrid fraud detection model combining classical neural network layers
with a parameterised Qiskit circuit for the fully‑connected quantum layer.

The model mirrors the structure of the original photonic implementation
but augments it with a discrete‑variable quantum circuit that can be
trained jointly with the classical parameters.  A custom autograd
function evaluates the circuit on an Aer simulator and returns the
expectation value of the computational‑basis measurement.  Clipping
logic is preserved so that the parameters remain within a fixed range
during optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from torch.autograd import Function


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


class _QuantumCircuit(Function):
    """PyTorch autograd wrapper around a Qiskit circuit."""

    @staticmethod
    def forward(ctx, theta: torch.Tensor, circuit: qiskit.QuantumCircuit,
                backend: Aer.AerSimulator, shots: int):
        bound = circuit.bind_parameters({circuit.parameters[0]: theta.item()})
        job = execute(bound, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound)
        probs = np.array([counts.get(bit, 0) for bit in sorted(counts)]) / shots
        bits = np.array([int(bit, 2) for bit in sorted(counts)])
        expectation = np.sum(bits * probs)
        ctx.save_for_backward(theta)
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots
        return torch.tensor(expectation, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        eps = 1e-3
        plus = _QuantumCircuit.apply(theta + eps, ctx.circuit, ctx.backend, ctx.shots)
        minus = _QuantumCircuit.apply(theta - eps, ctx.circuit, ctx.backend, ctx.shots)
        grad_theta = (plus - minus) / (2 * eps)
        return grad_output * grad_theta, None, None, None


class QuantumFullyConnectedLayer(nn.Module):
    """A single‑qubit variational circuit that acts like a fully‑connected layer."""

    def __init__(self, shots: int = 1024):
        super().__init__()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(theta, 0)
        self.circuit.measure_all()
        self.theta = nn.Parameter(torch.tensor(0.0))

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        theta_value = float(thetas[0]) if isinstance(thetas, (list, tuple)) else float(thetas)
        return _QuantumCircuit.apply(torch.tensor(theta_value, device=self.theta.device),
                                      self.circuit, self.backend, self.shots)


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection network with classical and quantum layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        classical_layers: Iterable[FraudLayerParameters],
        quantum_layer: QuantumFullyConnectedLayer,
        final_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_layer = _layer_from_params(input_params, clip=False)
        self.classical = nn.Sequential(
            *[_layer_from_params(p, clip=True) for p in classical_layers]
        )
        self.quantum = quantum_layer
        self.final = nn.Linear(2, 1)
        self.final.weight.data.fill_(final_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.classical(out)
        theta = out[:, 0]
        q_out = self.quantum(theta)
        out = torch.cat([out, q_out.unsqueeze(1)], dim=1)
        out = self.final(out)
        return out


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    classical_layers: Iterable[FraudLayerParameters],
    quantum_shots: int = 1024,
) -> FraudDetectionHybrid:
    """Convenience constructor that wires the quantum and classical parts."""
    quantum_layer = QuantumFullyConnectedLayer(shots=quantum_shots)
    return FraudDetectionHybrid(input_params, classical_layers, quantum_layer)


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
    "build_fraud_detection_model",
    "QuantumFullyConnectedLayer",
]
