"""Quantum hybrid self‑attention + fraud‑detection module.

The class builds a Qiskit circuit that first implements a
self‑attention style block and then appends a sequence of
parameterized gates inspired by the photonic fraud‑detection
layers.  The circuit is executed on a backend and returns the
measurement counts.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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

def _apply_fraud_layer(circuit: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    """Add a fraud‑detection style block to the circuit."""
    for i in range(circuit.num_qubits - 1):
        circuit.crx(_clip(params.bs_theta, 5.0) if clip else params.bs_theta, i, i + 1)
    for i, phase in enumerate(params.phases):
        circuit.rz(_clip(phase, 5.0) if clip else phase, i)
    for i, r in enumerate(params.squeeze_r):
        circuit.ry(_clip(r, 5.0) if clip else r, i)
    for i, r in enumerate(params.displacement_r):
        circuit.rx(_clip(r, 5.0) if clip else r, i)
    for i in range(circuit.num_qubits - 1):
        circuit.crz(_clip(params.kerr[0], 1.0) if clip else params.kerr[0], i, i + 1)

class SelfAttentionFraudDetector:
    """Quantum circuit implementing self‑attention + fraud‑detection."""
    def __init__(self, n_qubits: int, fraud_params: Iterable[FraudLayerParameters]):
        self.n_qubits = n_qubits
        self.fraud_params = list(fraud_params)

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Self‑attention block
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Fraud‑detection layers
        for params in self.fraud_params:
            _apply_fraud_layer(circuit, params, clip=True)

        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

backend = qiskit.Aer.get_backend("qasm_simulator")
attention = SelfAttentionFraudDetector(n_qubits=4, fraud_params=[])
__all__ = ["SelfAttentionFraudDetector", "FraudLayerParameters"]
