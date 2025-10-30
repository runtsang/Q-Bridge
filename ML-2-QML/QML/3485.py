from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑style fraud layer for the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionHybrid:
    """Quantum hybrid fraud‑detection circuit combining photonic‑style operations and a self‑attention block."""
    def __init__(self,
                 fraud_params: FraudLayerParameters,
                 attention_rotation: np.ndarray,
                 attention_entangle: np.ndarray):
        self.fraud_params = fraud_params
        self.attention_rotation = attention_rotation  # shape (12,)
        self.attention_entangle = attention_entangle  # shape (3,)
        self.backend = Aer.get_backend("qasm_simulator")

    def _photonic_subcircuit(self, qc: QuantumCircuit, qubits):
        fp = self.fraud_params
        # BS‑like rotations
        qc.rx(fp.bs_theta, qubits[0]); qc.rx(fp.bs_phi, qubits[1])
        # Phase gates
        qc.rz(fp.phases[0], qubits[0]); qc.rz(fp.phases[1], qubits[1])
        # Squeeze + displacement
        qc.rx(fp.squeeze_r[0], qubits[0]); qc.rz(fp.squeeze_phi[0], qubits[0])
        qc.rx(fp.squeeze_r[1], qubits[1]); qc.rz(fp.squeeze_phi[1], qubits[1])
        qc.rx(fp.displacement_r[0], qubits[0]); qc.rz(fp.displacement_phi[0], qubits[0])
        qc.rx(fp.displacement_r[1], qubits[1]); qc.rz(fp.displacement_phi[1], qubits[1])
        # Kerr‑like
        qc.rz(fp.kerr[0], qubits[0]); qc.rz(fp.kerr[1], qubits[1])

    def _attention_subcircuit(self, qc: QuantumCircuit, qubits):
        # rotation params
        for i in range(4):
            idx = 3 * i
            qc.rx(self.attention_rotation[idx], qubits[i])
            qc.ry(self.attention_rotation[idx + 1], qubits[i])
            qc.rz(self.attention_rotation[idx + 2], qubits[i])
        # entangle params
        for i in range(3):
            qc.crx(self.attention_entangle[i], qubits[i], qubits[i + 1])

    def build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        qc = QuantumCircuit(qr, cr)
        # Photonic subcircuit on first two qubits
        self._photonic_subcircuit(qc, qr[0:2])
        # Self‑attention subcircuit on all four qubits
        self._attention_subcircuit(qc, qr[0:4])
        qc.measure(qr, cr)
        return qc

    def run(self, shots: int = 1024):
        qc = self.build_circuit()
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
