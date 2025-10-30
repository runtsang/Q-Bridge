"""Quantum‑enhanced graph classifier helper.

The quantum module implements the same interface as the classical
``build_graph_classifier`` but uses a variational circuit that
operates on a number of qubits equal to the *embedding dimension*.
It also produces a fidelity‑based adjacency graph from the
output states of the circuit, enabling graph‑aware quantum loss
functions.  The circuit is deliberately simple yet expressive
and can be executed on a simulator or a real device via Qiskit.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import networkx as nx
import numpy as np
import scipy as sc


def _tensored_zero(num_qubits: int) -> np.ndarray:
    """Return a zero state |0>^{⊗n} as a NumPy array."""
    return np.zeros((2**num_qubits, 1), dtype=complex)


def _apply_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Compute |⟨a|b⟩|²."""
    return float(np.abs(state_a.conj().T @ state_b)[0, 0] ** 2)


def _create_fidelity_graph(
    states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None
) -> nx.Graph:
    """Build a graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = _apply_fidelity(a, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary)
    return graph


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a quantum variational circuit and return metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the embedding dimension of the classical GNN.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Variational ansatz with data‑encoding and entangling gates.
    encoding : Iterable[ParameterVector]
        Parameter vector for data encoding.
    weights : Iterable[ParameterVector]
        Parameter vector for variational weights.
    observables : List[SparsePauliOp]
        Pauli Z observables on each qubit for measurement.
    """

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangle adjacent qubits
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Measurement observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
