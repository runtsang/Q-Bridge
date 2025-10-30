"""Quantum classifier that mirrors the classical architecture and uses the
fidelity‑based adjacency to set qubit connectivity."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_quantum_classifier(
    num_qubits: int,
    depth: int,
    graph_edges: List[Tuple[int, int, float]],
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Build a variational circuit with data encoding, depth‑wise Ry/CZ layers,
    and Pauli‑Z observables.  The CZ connections are induced by the
    adjacency graph derived from the classical model.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the feature dimensionality).
    depth : int
        Number of variational layers.
    graph_edges : list of (i, j, weight)
        Edges from the fidelity adjacency graph; only the topology
        (i, j) is used to add CZ gates.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled circuit.
    encoding : list[ParameterVector]
        Parameters for data encoding (Rx gates).
    weights : list[ParameterVector]
        Variational parameters for Ry gates.
    observables : list[SparsePauliOp]
        Measurement operators (Z on each qubit).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding with Rx
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # CZ gates according to the graph topology
        for i, j, _ in graph_edges:
            circuit.cz(i, j)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables

__all__ = ["build_quantum_classifier"]
