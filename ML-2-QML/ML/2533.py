"""Hybrid fraud detection model combining classical neural network layers with a quantum feature extractor.

The model mirrors the photonic architecture of the original seed but augments it with a
parameterised quantum circuit implemented via Qiskit.  The classical part is a
sequence of fully‑connected layers derived from ``FraudLayerParameters``.  The
quantum part produces a single expectation value that is concatenated to the
classical output before the final regression head.

The design demonstrates how a classical network can be enriched with a
quantum sub‑module while keeping the training pipeline fully classical.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import Aer, execute
from dataclasses import dataclass
from typing import Iterable, Sequence

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
    activation = nn.ReLU()
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

class QuantumFeatureExtractor(nn.Module):
    """A tiny Qiskit‑based feature extractor used inside the hybrid model."""

    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs shape (batch, 1)
        thetas = inputs.squeeze().detach().cpu().numpy()
        expectations = []
        for theta in thetas:
            bound = {self.theta: theta}
            job = execute(
                self.circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=[bound],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectations.append(np.sum(states * probs))
        return torch.tensor(expectations, dtype=torch.float32).unsqueeze(1)

class FraudDetectionHybrid(nn.Module):
    """Hybrid classical‑quantum fraud‑detection model."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.classical = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(p, clip=True) for p in layers),
            nn.Linear(2, 1),
        )
        qp = quantum_params or {}
        self.quantum = QuantumFeatureExtractor(**qp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_out = self.classical(x)
        qfeat = self.quantum(x)
        return torch.cat([class_out, qfeat], dim=1)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
