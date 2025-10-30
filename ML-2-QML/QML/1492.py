"""Hybrid quantum Graph Neural Network using PennyLane.

Key additions:
* Variational circuit to produce a random target unitary.
* Random layer construction using qml.QubitUnitary gates.
* State propagation via PennyLane state vectors.
* Fidelity‑based adjacency remains unchanged.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

def _random_unitary_matrix(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix via QR decomposition."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q

def _random_qubit_unitary(num_qubits: int):
    """Return a PennyLane QubitUnitary gate with a random unitary."""
    return qml.QubitUnitary(_random_unitary_matrix(num_qubits))

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    return vec / np.linalg.norm(vec)

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate state pairs (|ψ⟩, U|ψ⟩)."""
    data: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        psi = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        data.append((psi, unitary @ psi))
    return data

def random_network(qnn_arch: List[int], samples: int):
    """Generate a random QNN architecture and training data."""
    # target unitary for the last layer
    target_unitary = _random_unitary_matrix(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qml.Operation]] = [[]]
    for layer in range(1, len(qnn_arch)):
        n_in = qnn_arch[layer - 1]
        n_out = qnn_arch[layer]
        layer_ops: List[qml.Operation] = []
        for _ in range(n_out):
            op = _random_qubit_unitary(n_in + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace(state: np.ndarray, keep: List[int]) -> np.ndarray:
    """Partial trace over all qubits not in `keep`."""
    dim = state.shape[0]
    num_qubits = int(np.log2(dim))
    # build density matrix
    rho = np.outer(state, state.conj())
    # trace out qubits not in keep
    for q in reversed(range(num_qubits)):
        if q not in keep:
            rho = np.trace(rho.reshape([2]*4, order='F'), axis1=2*q, axis2=2*q+1)
    return rho

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.Operation]],
                   layer: int, input_state: np.ndarray) -> np.ndarray:
    """Apply the layer's unitary and partial‑trace to keep the output qubits."""
    n_in = qnn_arch[layer - 1]
    n_out = qnn_arch[layer]
    # prepend input state with zeros for the output qubits
    zeros = np.zeros(2 ** n_out, dtype=complex)
    state = np.kron(input_state, zeros)
    # compose all gates
    for gate in unitaries[layer]:
        U = gate.matrix
        state = U @ state
    # partial trace to remove the input qubits
    return np.trace(state.reshape([2]*4, order='F'), axis1=0, axis2=1)

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.Operation]],
                samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Propagate each sample through the QNN."""
    outputs: List[List[np.ndarray]] = []
    for psi, _ in samples:
        layerwise = [psi]
        current = psi
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
