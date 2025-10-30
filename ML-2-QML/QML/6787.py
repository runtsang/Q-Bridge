"""Hybrid fraud‑detection model – quantum implementation using Strawberry Fields and Qiskit.

This module mirrors the classical `FraudDetectionHybrid` interface but replaces
the PyTorch layers with fully quantum sub‑circuits.  Photonic layers are built
with Strawberry Fields, while a Qiskit circuit emulates a fully connected
quantum layer.  The public API remains identical so that the same client code
can switch between classical and quantum back‑ends by importing from the
respective module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def build_photonic_layer(q, params: FraudLayerParameters, clip: bool) -> None:
    """Apply a photonic layer to the Strawberry Fields context."""
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]


def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        build_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            build_photonic_layer(q, layer, clip=True)
    return program


class QuantumFullyConnected:
    """A simple Qiskit circuit acting as a fully connected quantum layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 200):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        bound = {self.theta: float(next(thetas))}
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bound],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()), dtype=int)
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])


class FraudDetectionHybridQuantum:
    """Quantum‑centric counterpart of the classical hybrid model."""
    def __init__(
        self,
        photonic_layers: List[FraudLayerParameters],
        qiskit_thetas: Iterable[float] | None = None,
    ) -> None:
        self.photonic_program = build_photonic_program(
            input_params=photonic_layers[0],
            layers=photonic_layers[1:],
        )
        self.quantum_fc = QuantumFullyConnected() if qiskit_thetas else None
        self.qiskit_thetas = iter(qiskit_thetas) if qiskit_thetas else None

    def run_photonic(self, state: np.ndarray) -> np.ndarray:
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(self.photonic_program, args={"psi": state})
        return result.statevector()

    def run_quantum_fc(self) -> np.ndarray:
        if self.quantum_fc is None:
            raise RuntimeError("Quantum fully connected layer not configured.")
        return self.quantum_fc.run(self.qiskit_thetas)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        """Forward pass combining photonic and qubit sub‑circuits."""
        photonic_out = self.run_photonic(state)
        if self.quantum_fc:
            quantum_out = self.run_quantum_fc()
            # Concatenate outcomes as a simple feature vector
            return np.concatenate([photonic_out, quantum_out])
        return photonic_out


__all__ = ["FraudLayerParameters", "build_photonic_program", "QuantumFullyConnected", "FraudDetectionHybridQuantum"]
