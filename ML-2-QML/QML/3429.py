"""Quantum counterpart to the hybrid fraud detection model.

The module contains a graph‑based quantum neural network (QNN) that can be
used in tandem with the classical part.  All heavy quantum objects are
defined using qutip; the module is fully importable and provides a
clean API for generating random networks, training data, forward
propagation and fidelity‑based graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import itertools
import networkx as nx
import qutip as qt
import scipy as sc


# --------------------------------------------------------------------------- #
# 1. Fundamental Qubit helpers
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator on `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    identity.dims = [[2] * num_qubits, [2] * num_qubits]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Projector onto the all‑zero computational basis state for `num_qubits`."""
    projector = qt.fock(2 ** num_qubits).proj()
    projector.dims = [[2] * num_qubits, [2] * num_qubits]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Permute the qubit registers of the operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Sample a random Haar‑distributed unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


# --------------------------------------------------------------------------- #
# 2. Data generation utilities
# --------------------------------------------------------------------------- #
def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Return a list of (input_state, target_state) pairs for a target unitary."""
    dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Generate a random QNN with the given architecture.

    Returns:
        qnn_arch (list[int]): Layer widths.
        unitaries (list[list[qt.Qobj]]): One list of gates per layer.
        training_data (list[tuple[qt.Qobj, qt.Qobj]]): Input–output pairs.
        target_unitary (qt.Qobj): The final layer unitary that the network should approximate.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]  # first entry is dummy to keep indexing 1‑based
    for layer_idx in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer_idx - 1]
        num_outputs = qnn_arch[layer_idx]
        layer_ops: list[qt.Qobj] = []

        for out_idx in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)

            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1),
                               _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + out_idx)

            layer_ops.append(op)

        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


# --------------------------------------------------------------------------- #
# 3. Forward propagation
# --------------------------------------------------------------------------- #
def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Keep only the qubits specified in ``keep``."""
    return state.ptrace(list(keep))


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Remove the qubits specified in ``remove``."""
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a single layer of the QNN to ``input_state``."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Prepare the joint state: input tensor product with |0>^num_outputs
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    # Compose all gates in the layer
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    # Apply unitary and partial‑trace to drop the extra outputs
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> list[list[qt.Qobj]]:
    """Propagate each sample through the QNN and return all intermediate states."""
    stored_states: list[list[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# 4. State fidelity and graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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
# 5. Convenience facade
# --------------------------------------------------------------------------- #
class FraudGraphHybrid:
    """Facade for the quantum graph‑based neural network.

    The class simply forwards to the module‑level helpers to keep the API
    consistent with the classical counterpart.  This allows experiment code
    to import ``FraudGraphHybrid`` from either module and access the
    appropriate functionality.
    """
    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                    samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
        return feedforward(qnn_arch, unitaries, samples)

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           secondary: float | None = None, secondary_weight: float = 0.5):
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

__all__ = [
    "FraudGraphHybrid",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
