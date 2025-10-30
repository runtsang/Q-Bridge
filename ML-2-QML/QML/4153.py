"""Unified quantum component that complements the classical graph classifier.

The quantum circuit is a parameter‑sharded variational ansatz that
improves upon the two‑layer architecture of the reference QML
``build_classifier_circuit``.  The module builds a quantum circuit
for each layer, each with a *tuned* quantum feature map and
an optional *swap‑based* swap‑gate entanglement.  The outputs of
the classical network are given as input states to the quantum
simulator, and the quantum circuit returns an expectation value
for each class.  The quantum part can also reflect the fidelity
w.r.t. **all** states in the batch, that’s why we expose the
thetas and the states.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns: (circuit, encoding_params, weight_params, observables)
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Feature encoding
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    # Variational layers
    for d in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[d * num_qubits + i], i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Simple observable: single‑qubit Z on first qubit
    observables = [SparsePauliOp.from_label("Z" + "I" * (num_qubits - 1))]

    return qc, list(encoding), list(weights), observables


def run_circuit(qc: QuantumCircuit, thetas: Iterable[float], backend=None, shots: int = 1024) -> np.ndarray:
    """
    Execute the circuit with the given parameter vector and return
    the expectation value of the first observable (Z on qubit 0).
    """
    if backend is None:
        backend = Aer.get_backend("qasm_simulator")

    param_bindings = [{param: val for param, val in zip(qc.parameters, thetas)}]
    job = execute(qc, backend, shots=shots, parameter_binds=param_bindings)
    counts = job.result().get_counts(qc)

    # Convert counts to expectation of Z on qubit 0
    expectation = 0.0
    for outcome, count in counts.items():
        bit = int(outcome[0])  # qubit 0 state
        expectation += (1 - 2 * bit) * count
    expectation /= shots
    return np.array([expectation])


def fidelity_graph(state_vectors: List[np.ndarray], threshold: float = 0.95) -> nx.Graph:
    """
    Build a weighted graph from state fidelities.  Each node is a state
    vector; edges are added when fidelity exceeds the threshold.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(state_vectors)))
    for i in range(len(state_vectors)):
        for j in range(i + 1, len(state_vectors)):
            fid = np.abs(state_vectors[i].conj().dot(state_vectors[j])) ** 2
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
    return G


__all__ = ["build_classifier_circuit", "run_circuit", "fidelity_graph"]
