"""Quantum classifier circuit builder with angle encoding and entangling layers.

The `build_classifier_circuit` function builds a parameter‑shiftable ansatz
with explicit RX data encoding, CX‑based entanglement, and a variational
Ry layer.  It returns the circuit, encoding parameters, variational
parameters, and a list of `SparsePauliOp` observables for binary
classification.

Example
-------
>>> circ, enc, theta, obs = build_classifier_circuit(num_qubits=4, depth=2)
>>> print(circ)
QuantumCircuit(4)
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz for a quantum classifier.

    Parameters
    ----------
    num_qubits:
        Number of qubits (features).
    depth:
        Number of variational layers.

    Returns
    -------
    circuit:
        A `QuantumCircuit` ready for simulation or execution.
    encoding:
        List of data‑encoding parameters.
    weights:
        List of variational parameters.
    observables:
        List of Pauli operators to measure class scores.
    """
    # Data encoding: RX rotations
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters: one Ry per qubit per layer
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Apply data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Build variational layers
    w_idx = 0
    for _ in range(depth):
        # Variational rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[w_idx], qubit)
            w_idx += 1
        # Entanglement via CX in a ring topology
        for qubit in range(num_qubits):
            circuit.cx(qubit, (qubit + 1) % num_qubits)

    # Observables: Z on each qubit (binary classification via majority vote)
    observables = [
        SparsePauliOp("Z" + "I" * (num_qubits - 1)),
        SparsePauliOp("I" * (num_qubits - 1) + "Z"),
    ]
    return circuit, list(encoding), list(weights), observables
