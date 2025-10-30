"""Quantum fraud detection circuit mirroring the classical architecture.

Uses Qiskit to encode features, apply a self‑attention style entanglement,
and a variational ansatz.  The circuit can be executed on any Aer or
real backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit import Aer, execute


@dataclass
class FraudLayerParameters:
    """Placeholder for photonic‑style parameters (not used in the quantum circuit)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def build_quantum_fraud_circuit(num_qubits: int,
                                depth: int,
                                encoding_params: np.ndarray | None = None,
                                variational_params: np.ndarray | None = None) -> QuantumCircuit:
    """
    Build a quantum circuit that encodes input features, applies
    self‑attention style entanglement, and a variational depth‑controlled ansatz.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encoding: RX rotations parameterized by input features
    if encoding_params is None:
        encoding_params = ParameterVector("x", num_qubits)
    for i, param in enumerate(encoding_params):
        circuit.rx(param, i)

    # Self‑attention style entanglement: CRX gates
    for i in range(num_qubits - 1):
        circuit.crx(encoding_params[i], i, i + 1)

    # Variational layers
    if variational_params is None:
        variational_params = ParameterVector("theta", num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(variational_params[idx], q)
            idx += 1
        # Entanglement: CZ between neighboring qubits
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Measurement in Z basis
    circuit.measure(qr, cr)
    return circuit


def run_quantum_fraud(circuit: QuantumCircuit,
                      backend=None,
                      shots: int = 1024) -> dict:
    """
    Execute the circuit on the specified backend and return outcome counts.
    """
    if backend is None:
        backend = Aer.get_backend("qasm_simulator")
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    return result.get_counts(circuit)


__all__ = ["FraudLayerParameters", "build_quantum_fraud_circuit",
           "run_quantum_fraud"]
