"""Quantum graph‑neural‑network with a parameter‑shaped variational ansatz.

This module implements the same API as the classical version but
uses Qiskit to construct a layered unitary that operates on a
graph‑structured set of qubits.  The circuit is built from
parameterized single‑qubit rotations and CNOT entanglers that
mirror the topology of the input graph.  The `sample_counts`
helper allows measurement statistics to be extracted for downstream
analysis.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Sequence

import networkx as nx
import numpy as np
import qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

Tensor = qiskit.quantum_info.Statevector


# --------------------------------------------------------------------------- #
# 1.  Utility functions
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> qiskit.quantum_info.Unitary:
    """Generate a Haar‑random unitary on ``num_qubits``."""
    return qi.random_unitary(2 ** num_qubits)


def _random_state(num_qubits: int) -> qiskit.quantum_info.Statevector:
    """Create a random pure state on ``num_qubits``."""
    vec = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
    vec /= np.linalg.norm(vec)
    return qi.Statevector(vec)


def random_training_data(unitary: qiskit.quantum_info.Unitary, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (state, U|state>)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.dim[0]
    for _ in range(samples):
        state = _random_state(int(np.log2(dim)))
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return an architecture, a list of parameter vectors for each layer,
    a training set, and the target unitary to be learned."""
    num_layers = len(qnn_arch) - 1
    params: List[ParameterVector] = []
    for layer in range(num_layers):
        # each layer gets a rotation for each qubit in that layer
        params.append(ParameterVector(f"θ_{layer}", qnn_arch[layer]))
    # target unitary is random
    target = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target, samples)
    return list(qnn_arch), params, training_data, target


def _layer_circuit(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
    """Build a single layer of the ansatz: RY rotations + CNOT chain."""
    qc = QuantumCircuit(num_qubits)
    # single‑qubit rotations
    for q in range(num_qubits):
        qc.ry(params[q], q)
    # entanglement via CNOT chain
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def _apply_layer(state: Tensor, qc: QuantumCircuit) -> Tensor:
    """Apply a layer circuit to a state vector."""
    unitary = qi.Operator(qc)
    return unitary @ state


# --------------------------------------------------------------------------- #
# 2.  Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[ParameterVector],
    samples: Iterable[Tuple[Tensor, Tensor]],
    param_values: Sequence[np.ndarray] | None = None,
) -> List[List[Tensor]]:
    """
    Forward propagate each sample through the variational circuit.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the QNN (number of qubits per layer).
    params : Sequence[ParameterVector]
        List of parameter vectors for each layer.
    samples : Iterable[Tuple[Tensor, Tensor]]
        Training samples (state, target).
    param_values : Sequence[np.ndarray] | None
        Optional concrete parameter values; if omitted, the symbolic
        circuit is kept.

    Returns
    -------
    List[List[Tensor]]
        A list of state vectors for each layer per sample.
    """
    stored: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise: List[Tensor] = [state]
        current = state
        for layer_idx, param_vec in enumerate(params):
            qc = _layer_circuit(qnn_arch[layer_idx], param_vec)
            if param_values is not None:
                binding = dict(zip(param_vec, param_values[layer_idx]))
                qc.assign_parameters(binding, inplace=True)
            current = _apply_layer(current, qc)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


# --------------------------------------------------------------------------- #
# 3.  State fidelity and graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4.  Measurement helper
# --------------------------------------------------------------------------- #
def sample_counts(states: Sequence[Tensor], shots: int = 1024) -> List[dict[str, int]]:
    """Return measurement counts for each state vector."""
    backend = Aer.get_backend("qasm_simulator")
    results: List[dict[str, int]] = []
    for state in states:
        qc = state.to_circuit()
        qc.measure_all()
        job = execute(qc, backend=backend, shots=shots)
        counts = job.result().get_counts()
        results.append(counts)
    return results


# --------------------------------------------------------------------------- #
# 5.  Convenience class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Wrapper around the variational circuit that builds a Qiskit
    circuit for a given architecture and parameter set.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture (number of qubits per layer).
    params : Sequence[ParameterVector]
        Parameter vectors for each layer.
    """

    def __init__(self, qnn_arch: Sequence[int], params: Sequence[ParameterVector]):
        self.arch = list(qnn_arch)
        self.params = params

    def build_circuit(self, param_values: Sequence[np.ndarray] | None = None) -> QuantumCircuit:
        """Build a full circuit with optional concrete parameters."""
        total_qubits = self.arch[-1]
        qc = QuantumCircuit(total_qubits)
        for layer_idx, param_vec in enumerate(self.params):
            layer_qc = _layer_circuit(self.arch[layer_idx], param_vec)
            if param_values is not None:
                binding = dict(zip(param_vec, param_values[layer_idx]))
                layer_qc.assign_parameters(binding, inplace=True)
            qc.append(layer_qc, range(total_qubits))
        return qc

    def sample_counts(self, shots: int = 1024) -> List[dict[str, int]]:
        """Return measurement counts for the target state."""
        circuit = self.build_circuit()
        circuit.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend=backend, shots=shots)
        return job.result().get_counts()

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
