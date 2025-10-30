"""Hybrid quantum classifier with encoding, attention‑style block, ansatz, and measurements."""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a quantum classifier that combines data encoding, a quantum self‑attention block, and a variational ansatz."""
    encoding = ParameterVector("x", num_qubits)
    rotation_params = ParameterVector("theta", num_qubits * 3)   # rx,ry,rz per qubit
    entangle_params = ParameterVector("phi", num_qubits - 1)     # controlled rotations
    weights = ParameterVector("w", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    # Quantum attention‑style block
    for i in range(num_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(rotation_params) + list(entangle_params) + list(weights), observables

__all__ = ["build_classifier_circuit"]
