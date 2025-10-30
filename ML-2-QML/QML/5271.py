"""Quantum components for fraud detection: photonic circuit builder, quantum kernel, and quanvolution filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
#   Photonic‑inspired quantum circuit
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Container for a single photonic layer's parameters."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> tq.QuantumDevice:
    """Create a quantum device executing the photonic‑inspired circuit."""
    device = tq.QuantumDevice(n_wires=2)
    with device:
        _apply_layer(device, input_params, clip=False)
        for layer in layers:
            _apply_layer(device, layer, clip=True)
    return device


def _apply_layer(device: tq.QuantumDevice, params: FraudLayerParameters, clip: bool) -> None:
    device.bswap(params.bs_theta, params.bs_phi)
    for i, phase in enumerate(params.phases):
        device.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        device.s(r if not clip else _clip(r, 5), phi, i)
    device.bswap(params.bs_theta, params.bs_phi)
    for i, phase in enumerate(params.phases):
        device.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        device.d(r if not clip else _clip(r, 5), phi, i)
    for i, k in enumerate(params.kerr):
        device.k(k if not clip else _clip(k, 1), i)


# --------------------------------------------------------------------------- #
#   Quantum kernel via TorchQuantum
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Encode two‑dimensional data via Ry rotations and a fixed entangling layer."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = [
            {"func": "ry", "wires": [i], "input_idx": [i]} for i in range(n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluating overlap of two data vectors."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#   Quanvolution filter via Qiskit
# --------------------------------------------------------------------------- #
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class QuanvCircuit:
    """Quantum filter emulating a 2×2 quanvolution kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 1024) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.threshold = threshold
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {theta: (np.pi if val > self.threshold else 0) for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        counts = job.result().get_counts(self.circuit)
        ones = sum(int(bit) for key in counts for bit in key) * list(counts.values())[0]
        return ones / (self.shots * self.n_qubits)


def Conv() -> QuanvCircuit:
    return QuanvCircuit(kernel_size=2, threshold=0.5, shots=1024)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "Kernel", "kernel_matrix", "QuanvCircuit", "Conv"]
