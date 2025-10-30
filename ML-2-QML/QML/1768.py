"""Utilities for building graph-based quantum neural networks with Pennylane.

The module retains the forward propagation helpers and the
fidelity-based adjacency construction used in the original QML
implementation while replacing the heavy quTiP dependencies with
Pennylane.  A variational circuit is constructed layer‑by‑layer
with parameterised `qml.Rot` gates.  The `feedforward` function returns
the sequence of quantum states produced after each layer, which can be
used to build a fidelity‑based graph."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pennylane as qml
import numpy as np
import scipy as sc


# --------------------------------------------------------------------------- #
#  Core QNN state propagation helpers
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.QubitUnitary:
    return qml.Identity(num_qubits)


def _tensored_zero(num_qubits: int) -> qml.QubitUnitary:
    return qml.QubitUnitary(np.zeros((2 ** num_qubits, 2 ** num_qubits)))


def _swap_registers(op: qml.Operation, source: int, target: int) -> qml.Operation:
    if source == target:
        return op
    order = list(range(len(op.wires)))
    order[source], order[target] = order[target], order[source]
    return qml.apply(op, wires=order)


def _random_qubit_unitary(num_qubits: int) -> qml.QubitUnitary:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    return qml.QubitUnitary(unitary)


def _random_qubit_state(num_qubits: int) -> qml.StateVector:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    return qml.StateVector(amplitudes)


# --------------------------------------------------------------------------- #
#  Training data generation
# --------------------------------------------------------------------------- #
def random_training_data(unitary: qml.QubitUnitary, samples: int) -> List[Tuple[qml.StateVector, qml.StateVector]]:
    dataset: List[Tuple[qml.StateVector, qml.StateVector]] = []
    num_qubits = unitary.wires
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qml.Operation]], List[Tuple[qml.StateVector, qml.StateVector]], qml.QubitUnitary]:
    """Create a random variational circuit and generate training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qml.Operation]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qml.Operation] = []
        for output in range(num_outputs):
            # Each output qubit gets a parameterised Rot gate
            params = np.random.uniform(-np.pi, np.pi, size=3)
            gate = qml.Rot(*params, wires=[num_inputs + output])
            layer_ops.append(gate)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


# --------------------------------------------------------------------------- #
#  Partial trace helpers (simple implementation)
# --------------------------------------------------------------------------- #
def _partial_trace_remove(state: qml.StateVector, remove: Sequence[int]) -> qml.StateVector:
    keep = [i for i in range(state.shape[0] // 2) if i not in remove]
    return qml.partial_trace(state, keep)


# --------------------------------------------------------------------------- #
#  Layer channel application
# --------------------------------------------------------------------------- #
def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.Operation]],
    layer: int,
    input_state: qml.StateVector,
) -> qml.StateVector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad the state with zeros for the output qubits
    state = qml.StateVector(np.kron(input_state.data, np.zeros(2 ** num_outputs)))
    layer_ops = unitaries[layer]
    for op in layer_ops:
        state = op @ state
    # Remove input qubits to keep only the output state
    return _partial_trace_remove(state, range(num_inputs))


# --------------------------------------------------------------------------- #
#  Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.Operation]],
    samples: Iterable[Tuple[qml.StateVector, qml.StateVector]],
) -> List[List[qml.StateVector]]:
    """Return the state at each layer for every sample."""
    stored_states: List[List[qml.StateVector]] = []
    for sample, _ in samples:
        layerwise: List[qml.StateVector] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
#  Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qml.StateVector, b: qml.StateVector) -> float:
    """Return the absolute squared overlap between pure states `a` and `b`."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qml.StateVector],
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
#  Demo / test harness
# --------------------------------------------------------------------------- #
def main():
    arch = [2, 3, 4]
    samples = 5
    qnn_arch, ops, training_data, target_unitary = random_network(arch, samples)
    states = [sample for sample, _ in training_data]
    stored = feedforward(qnn_arch, ops, training_data)
    final_states = [layer_states[-1] for layer_states in stored]
    G = fidelity_adjacency(final_states, 0.95)
    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())


if __name__ == "__main__":
    main()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
