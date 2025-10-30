"""Quantum circuit factory for fraud‑detection, mirroring the photonic‑style and variational designs.

The circuit:
* Encodes each feature via an RX gate.
* Adds `depth` variational layers of RY rotations followed by a linear CZ entanglement chain.
* Measures Z on every qubit; observables are returned as a list of SparsePauliOp objects.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz that mirrors the classical fraud‑detection architecture.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) to encode.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The full quantum circuit.
    encoding : list
        ParameterVector objects for the input encoding.
    weights : list
        ParameterVector objects for the variational parameters.
    observables : list[SparsePauliOp]
        Pauli‑Z observables for readout.
    """
    # Encoding of classical features via RX
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        # Variational RY layer
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling CZ chain
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
