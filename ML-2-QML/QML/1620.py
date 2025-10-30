"""Quantum graph neural network utilities using Pennylane.

This module replaces the Qutip implementation with a lightweight
Numpy/Pennylane approach.  It provides a variational circuit that
attempts to reproduce a target unitary on random states and returns
a fidelity graph describing the learned representation.
"""

import itertools
from typing import List, Tuple, Iterable, Sequence

import numpy as np
import pennylane as qml
import networkx as nx


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix via QR decomposition."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Sample a random pure state of the given qubit count."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create (input, target) pairs where target = unitary * input."""
    dataset = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random layered unitary network and its training data."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    # For each layer, create a random unitary that acts on the full space.
    # The number of qubits per layer is given by qnn_arch[i].
    unitaries: List[np.ndarray] = []
    for n_qubits in qnn_arch[1:]:
        unitaries.append(_random_qubit_unitary(n_qubits))

    return list(qnn_arch), unitaries, training_data, target_unitary


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Propagate each sample through the layered unitaries."""
    stored_states = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for unitary in unitaries:
            # If current dimension is smaller, pad with |0> state
            dim_current = current.shape[0]
            dim_target = unitary.shape[0]
            if dim_current < dim_target:
                pad = np.zeros(dim_target // dim_current - 1, dtype=complex)
                pad[0] = 1.0
                current = np.kron(current, pad)
            current = unitary @ current
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def train_qml_network(
    qnn_arch: Sequence[int],
    samples: int,
    epochs: int = 200,
    lr: float = 0.01,
    fidelity_threshold: float = 0.95,
    secondary_threshold: float | None = None,
) -> Tuple[np.ndarray, nx.Graph, List[np.ndarray]]:
    """
    Train a simple variational circuit to approximate the target unitary.
    The circuit consists of two rotation layers (RX, RZ) per qubit followed
    by a nearest‑neighbour CNOT chain.  Parameters are optimized to minimize
    the average state‑fidelity loss over the training set.
    """
    arch, _unused_units, training_data, target_unitary = random_network(qnn_arch, samples)
    num_qubits = arch[-1]

    dev = qml.device("default.qubit", wires=num_qubits)

    # Parameters: two angles per qubit (RX, RZ)
    params_shape = (num_qubits, 2)
    params = np.random.randn(*params_shape)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        for i in range(num_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
        # nearest‑neighbour CNOT chain
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    def loss_fn(params):
        loss = 0.0
        for state, tgt in training_data:
            circ_state = circuit(params)
            loss += 1 - state_fidelity(circ_state, tgt)
        return loss / len(training_data)

    grad_fn = qml.grad(loss_fn)

    for _ in range(epochs):
        grads = grad_fn(params)
        params -= lr * grads

    # Build fidelity graph from the final circuit outputs on the training set
    final_states = [circuit(params) for _, _ in training_data]
    graph = fidelity_adjacency(final_states, fidelity_threshold, secondary=secondary_threshold)

    return target_unitary, graph, final_states


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_qml_network",
]
