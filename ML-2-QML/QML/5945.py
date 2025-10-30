"""GraphQNN: Quantum variant with parameterized variational circuits and fidelity‑based training utilities.

The quantum version mirrors the classical API but replaces weight matrices with
parameterized unitary gates.  It supports:
* a simple variational circuit builder that schedules arbitrary two‑qubit gates
* a fidelity‑based loss that compares the circuit output with a target unitary
* a training routine that optimises gate parameters using a classical optimiser
* a helper that extracts measurement probabilities to build a state graph.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import scipy as sc

Tensor = np.ndarray
QObj = qml.QubitStateVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.Device:
    """Return a device with an identity operator on *num_qubits*."""
    return qml.device("default.qubit", wires=num_qubits)


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix for *num_qubits* qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    return unitary


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training data for a target unitary."""
    dataset = []
    for _ in range(samples):
        state = np.random.randn(2, 2 ** 1) + 1j * np.random.randn(2, 2 ** 1)
        state = state / np.linalg.norm(state)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit and training data for the final layer."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    gates: List[List[qml.Operation]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_gates: List[qml.Operation] = []
        for output in range(num_outputs):
            gate = qml.CNOT(wires=[output, output + 1])  # simple placeholder
            layer_gates.append(gate)
        gates.append(layer_gates)

    return list(qnn_arch), gates, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    gates: Sequence[Sequence[qml.Operation]],
    layer: int,
    input_state: QObj,
) -> QObj:
    """Apply a layer of gates to *input_state* and return the output state."""
    for gate in gates[layer]:
        gate.apply(input_state)
    return input_state


def feedforward(
    qnn_arch: Sequence[int],
    gates: Sequence[Sequence[qml.Operation]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[QObj]]:
    """Forward propagation of a variational circuit."""
    stored_states: List[List[QObj]] = []
    for sample, _ in samples:
        current_state = qml.QubitStateVector(sample, wires=list(range(len(sample))))
        layerwise = [current_state]
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, gates, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: QObj, b: QObj) -> float:
    """Return the absolute squared overlap between two pure state vectors."""
    return abs(np.vdot(a.state, b.state)) ** 2


def fidelity_adjacency(
    states: Sequence[QObj],
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
# Training routine
# --------------------------------------------------------------------------- #
def train(
    qnn_arch: Sequence[int],
    gates: Sequence[Sequence[qml.Operation]],
    training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
    epochs: int = 5,
    lr: float = 0.01,
    loss_fn: Callable[[np.ndarray, np.ndarray], float] = lambda out, tgt: np.mean((out - tgt) ** 2),
) -> Tuple[List[float], List[float]]:
    """Simple training loop for a variational circuit."""
    optimizer = qml.GradientDescentOptimizer(lr)
    loss_history: List[float] = []
    fid_history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for state, target in training_data:
            def cost():
                out = qml.apply(gates, state)
                return loss_fn(out, target)
            loss = optimizer.step(cost)
            epoch_loss += loss

        # Fidelity on a random sample
        state, target = next(iter(training_data))
        out = qml.apply(gates, state)
        fid = state_fidelity(out, target)
        loss_history.append(epoch_loss / len(training_data))
        fid_history.append(fid)

    return loss_history, fid_history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
]
