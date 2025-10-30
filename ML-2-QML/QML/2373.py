"""Quantum fraud‑detection circuit using Qiskit.

This module implements a parametric two‑qubit circuit that mirrors the
classical photonic fraud‑detection network.  Parameters are taken from
`FraudLayerParameters` and mapped to rotation angles and entangling
gates.  The circuit can be simulated with Qiskit's Aer simulator and
used in variational training.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter

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

class FraudQuantumCircuit:
    """Parametric two‑qubit quantum circuit for fraud detection."""

    def __init__(self, params: FraudLayerParameters) -> None:
        self.params = params

    def build_circuit(self, qubits: int = 2) -> QuantumCircuit:
        qr = QuantumRegister(qubits, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Entangling gate analogous to a beamsplitter
        circ.cx(qr[0], qr[1])
        circ.rz(self.params.bs_phi, qr[0])
        circ.rx(self.params.bs_theta, qr[1])

        # Phase rotations
        circ.rz(self.params.phases[0], qr[0])
        circ.rz(self.params.phases[1], qr[1])

        # Squeeze-like rotations
        circ.ry(self.params.squeeze_r[0], qr[0])
        circ.ry(self.params.squeeze_r[1], qr[1])
        circ.rz(self.params.squeeze_phi[0], qr[0])
        circ.rz(self.params.squeeze_phi[1], qr[1])

        # Displacement-like rotations
        circ.rx(self.params.displacement_r[0], qr[0])
        circ.rx(self.params.displacement_r[1], qr[1])
        circ.rz(self.params.displacement_phi[0], qr[0])
        circ.rz(self.params.displacement_phi[1], qr[1])

        # Kerr‑like controlled‑phase
        circ.cp(self.params.kerr[0], qr[0], qr[1])
        circ.cp(self.params.kerr[1], qr[1], qr[0])

        # Measurement
        circ.measure(qr[0], cr[0])
        return circ

    def simulate(self, circ: QuantumCircuit, shots: int = 1024) -> dict:
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circ, backend=backend, shots=shots)
        result = job.result()
        return result.get_counts(circ)

    def expectation_z(self, circ: QuantumCircuit, shots: int = 1024) -> float:
        counts = self.simulate(circ, shots)
        pos = counts.get("0", 0)
        neg = counts.get("1", 0)
        return (pos - neg) / shots

__all__ = [
    "FraudLayerParameters",
    "FraudQuantumCircuit",
]
