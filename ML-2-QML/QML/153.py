"""Quantum‑aware graph‑based QNN module that re‑implements the seed with a variational circuit.

This module adds:
1. A Pennylane variational circuit that generates a parameterised unitary.
2. Training routine that optimises the circuit parameters to match a target unitary.
3. A sparsity analysis that examines the probability distribution of intermediate states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import numpy as np

Tensor = np.ndarray


def _tensored_id(num_qubits: int) -> np.ndarray:
    """Return the identity matrix for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    return np.eye(dim, dtype=complex)


def _tensored_zero(num_qubits: int) -> np.ndarray:
    """Return the projector onto the all‑zero state for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0
    return np.outer(vec, vec.conj())


def _swap_registers(op: np.ndarray, source: int, target: int) -> np.ndarray:
    """Swap two qubits in a unitary matrix."""
    if source == target:
        return op
    num_qubits = int(np.log2(op.shape[0]))
    perm = list(range(num_qubits))
    perm[source], perm[target] = perm[target], perm[source]
    Q = np.array([list(bin(i)[2:].zfill(num_qubits)) for i in range(2**num_qubits)])
    new_indices = [int("".join(str(Q[i][j]) for j in perm), 2) for i in range(2**num_qubits)]
    return op[new_indices][:, new_indices]


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training data (state, target_state) pairs."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(int(np.log2(unitary.shape[0])))
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Build a random QNN architecture with variational layers."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Partial trace over all qubits not in ``keep``."""
    return qml.math.partial_trace(state, keep)


def _partial_trace_remove(state: np.ndarray, remove: Sequence[int]) -> np.ndarray:
    """Partial trace over qubits in ``remove``."""
    keep = list(range(int(np.log2(state.shape[0]))))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    layer: int,
    input_state: np.ndarray,
) -> np.ndarray:
    """Apply the layer unitary to the input state and trace out the added qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad input state with zeros for the output qubits
    zero_state = np.zeros(2 ** num_outputs, dtype=complex)
    state = np.concatenate([input_state, zero_state])
    # Compose the unitary
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    # Apply and trace out output qubits
    new_state = layer_unitary @ state
    return _partial_trace_remove(new_state, range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Run a forward pass through the QNN, returning intermediate states."""
    stored_states: List[List[np.ndarray]] = []
    for state, _ in samples:
        layerwise = [state]
        current_state = state
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
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
# NEW FEATURES: variational circuit and training
# --------------------------------------------------------------------------- #

def variational_circuit(params: np.ndarray, wires: Sequence[int]) -> np.ndarray:
    """Return the unitary matrix produced by a simple Pennylane variational circuit."""
    dev = qml.device("default.qubit", wires=wires)
    num_qubits = len(wires)

    @qml.qnode(dev, interface="numpy")
    def circuit():
        idx = 0
        for q in wires:
            qml.RX(params[idx], wires=q)
            idx += 1
        for q in range(num_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return circuit()


def train_variational(
    training_data: List[Tuple[np.ndarray, np.ndarray]],
    num_qubits: int,
    init_params: np.ndarray | None = None,
    lr: float = 0.01,
    epochs: int = 200,
) -> np.ndarray:
    """Train the variational circuit to match the target unitary on the training data."""
    if init_params is None:
        init_params = np.random.randn(num_qubits)
    params = init_params.copy()
    optimizer = qml.GradientDescentOptimizer(stepsize=lr)

    for _ in range(epochs):
        loss = 0.0
        for state, target in training_data:
            def loss_fn(p):
                pred = variational_circuit(p, range(num_qubits))
                return np.mean(np.abs(pred - target) ** 2)

            loss += loss_fn(params)
            params = optimizer.step(loss_fn, params)
        loss /= len(training_data)
    return params


# --------------------------------------------------------------------------- #
# NEW FEATURES: quantum sparsity analysis
# --------------------------------------------------------------------------- #

def quantum_sparsity_analysis(
    states: List[List[np.ndarray]],
    threshold: float = 0.1,
) -> nx.Graph:
    """Build a graph where nodes are layers and edges indicate similarity of probability sparsity."""
    layer_sparsity: List[float] = []
    for layer_index in range(len(states[0])):  # assume uniform depth
        probs = [np.abs(state[layer_index]) ** 2 for state in states]
        probs = np.concatenate(probs)
        sparsity = np.mean(probs < threshold)
        layer_sparsity.append(sparsity)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(layer_sparsity)))
    for i, s_i in enumerate(layer_sparsity):
        for j, s_j in enumerate(layer_sparsity):
            if i >= j:
                continue
            weight = 1.0 - abs(s_i - s_j)
            graph.add_edge(i, j, weight=weight)
    return graph


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "variational_circuit",
    "train_variational",
    "quantum_sparsity_analysis",
]
