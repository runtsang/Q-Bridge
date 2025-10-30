"""Quantum‑photonic fraud detection circuit using Qiskit and a quantum convolution filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import qiskit
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

class FraudDetectionQMLCircuit:
    """Photonic‑style fraud detection circuit for Qiskit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        kernel_size: int = 2,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.kernel_size = kernel_size

    def build_circuit(self, shots: int = 100) -> qiskit.QuantumCircuit:
        n = self.kernel_size ** 2
        qreg = qiskit.QuantumRegister(n)
        creg = qiskit.ClassicalRegister(n)
        circ = qiskit.QuantumCircuit(qreg, creg)

        self._apply_layer(circ, self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(circ, layer, clip=True)

        circ.measure(qreg, creg)
        return circ

    def _apply_layer(
        self,
        circ: qiskit.QuantumCircuit,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> None:
        # Beam‑splitter emulation – RX + RZ
        for idx, theta in enumerate([params.bs_theta, params.bs_phi]):
            circ.rx(theta, idx)
            circ.rz(theta, idx)

        # Phase shifts
        for idx, phase in enumerate(params.phases):
            circ.rz(phase, idx)

        # Squeezing emulation – S + RZ
        for idx, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            circ.s(idx)
            circ.rz(phi, idx)

        # Displacement emulation with RX
        for idx, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            circ.rx(r if not clip else _clip(r, 5.0), idx)

        # Kerr non‑linearity via small RZ
        for idx, k in enumerate(params.kerr):
            circ.rz(k if not clip else _clip(k, 1.0), idx)

    def run(
        self,
        shots: int = 100,
        backend: qiskit.providers.Backend | None = None,
    ) -> dict[str, int]:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        circ = self.build_circuit(shots=shots)
        job = qiskit.execute(circ, backend=backend, shots=shots)
        return job.result().get_counts(circ)

def Conv() -> "QuanvCircuit":
    """Quantum convolution filter used for quanvolution layers."""
    class QuanvCircuit:
        def __init__(
            self,
            kernel_size: int = 2,
            backend: qiskit.providers.Backend | None = None,
            shots: int = 100,
            threshold: float = 0.5,
        ) -> None:
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [
                qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
            ]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for row in data:
                bind = {
                    theta: np.pi if val > self.threshold else 0
                    for theta, val in zip(self.theta, row)
                }
                param_binds.append(bind)
            job = qiskit.execute(
                self._circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self._circuit)
            counts = sum(
                sum(int(b) for b in key) * val for key, val in result.items()
            )
            return counts / (self.shots * self.n_qubits)

    return QuanvCircuit()

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionQMLCircuit",
    "Conv",
]
