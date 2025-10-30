"""Hybrid binary classifier that replaces the classical head with a quantum variational layer and a quantum convolution filter."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Optional

import qiskit
from qiskit import Aer, execute, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

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

class QuantumConvFilter(nn.Module):
    """Quantum analogue of a 2‑D convolution using a random circuit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, kernel_size, kernel_size)
        batch = x.shape[0]
        probs = []
        for i in range(batch):
            vals = x[i, 0].flatten().numpy()
            param_binds = [{t: np.pi if v > self.threshold else 0 for t, v in zip(self.theta, vals)}]
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result()
            counts = result.get_counts(self.circuit)
            total = 0
            for key, val in counts.items():
                ones = sum(int(bit) for bit in key)
                total += ones * val
            probs.append(total / (self.shots * self.n_qubits))
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)
        return probs

class QuantumHybridLayer(nn.Module):
    """Variational quantum layer that outputs a single probability."""
    def __init__(self, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.shots = shots
        self.shift = shift
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1)
        batch = x.shape[0]
        probs = []
        for i in range(batch):
            param = x[i, 0].item()
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[{self.theta: param}])
            result = job.result()
            counts = result.get_counts(self.circuit)
            p0 = counts.get('0', 0) / self.shots
            probs.append(2 * p0 - 1)
        return torch.tensor(probs, dtype=torch.float32).unsqueeze(-1)

class HybridBinaryClassifier(nn.Module):
    """End‑to‑end hybrid quantum‑classical binary classifier."""
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
        shots: int = 100,
    ) -> None:
        super().__init__()
        self.conv = QuantumConvFilter(kernel_size=conv_kernel, threshold=conv_threshold, shots=shots)
        if fraud_params is not None:
            layers = list(fraud_params)
            self.fraud = build_fraud_detection_program(layers[0], layers[1:])
        else:
            self.fraud = nn.Identity()
        self.quantum_head = QuantumHybridLayer(shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch, 1, H, W)
        x = self.conv(x)
        x = self.fraud(x)
        x = self.quantum_head(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QuantumConvFilter",
    "QuantumHybridLayer",
    "HybridBinaryClassifier",
]
