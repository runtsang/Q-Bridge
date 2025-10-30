"""FraudDetectionHybrid: classical backbone for a hybrid fraud‑detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a single layer, matching the photonic design but with optional clipping."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a value to a symmetric interval."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear‑activation block that emulates a photonic layer."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Assemble a deep network that mirrors the photonic stack and ends with a binary head."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def conv_quantum_filter(kernel_size: int = 2,
                       shots: int = 100,
                       threshold: float = 127.0,
                       backend: str = "qasm_simulator") -> "QuanvCircuit":
    """Return a Qiskit quanvolution filter that processes a 2×2 block."""
    import numpy as np
    import qiskit
    from qiskit.circuit.random import random_circuit

    class QuanvCircuit:
        """Quantum filter used as a drop‑in replacement for Conv() in the hybrid stack."""
        def __init__(self, kernel_size: int, backend: qiskit.providers.Backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

            # Encode data into rotation angles
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)

            # Randomised entangling layer
            self._circuit += random_circuit(self.n_qubits, depth=2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            """Run the circuit on a 2×2 block and return the mean |1> probability."""
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self._circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    backend = qiskit.Aer.get_backend(backend)
    return QuanvCircuit(kernel_size, backend, shots, threshold)


class FraudDetectionHybrid:
    """Hybrid fraud‑detection model that bundles the classical backbone and a quantum filter."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127.0,
        backend: str = "qasm_simulator",
    ) -> None:
        self.classical_model = build_fraud_detection_program(input_params, layers)
        self.quantum_filter = conv_quantum_filter(kernel_size, shots, threshold, backend)


__all__ = ["FraudLayerParameters",
           "build_fraud_detection_program",
           "conv_quantum_filter",
           "FraudDetectionHybrid"]
