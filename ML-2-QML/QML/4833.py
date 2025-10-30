"""Quantum graph‑neural‑network module that mirrors GraphQNNML.

The class GraphQNNQML provides the same public API as the classical
counterpart but uses qutip unitaries for propagation and a
quanvolution filter for local feature extraction.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import qiskit
import qutip as qt
import scipy as sc
import networkx as nx

from.Conv import Conv

Tensor = qt.Qobj


def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNQML:
    """Quantum graph‑neural‑network mirroring GraphQNNML.

    The class uses qutip unitaries for propagation and a quanvolution filter
    for local feature extraction.  The public API matches the classical
    implementation, enabling drop‑in replacement and comparative study.
    """

    def __init__(
        self,
        arch: Sequence[int],
        adjacency_threshold: float = 0.8,
        secondary_threshold: float | None = None,
    ):
        self.arch = list(arch)
        self.adjacency_threshold = adjacency_threshold
        self.secondary_threshold = secondary_threshold
        self.conv = Conv()  # quantum convolution filter
        self.unitaries: List[List[qt.Qobj]] | None = None

    def build_graph(self, states: Sequence[qt.Qobj]) -> nx.Graph:
        """Create a graph from state‑fidelity of node states."""
        return fidelity_adjacency(
            states,
            self.adjacency_threshold,
            secondary=self.secondary_threshold,
        )

    def set_unitaries(self, unitaries: List[List[qt.Qobj]]):
        """Store the layer‑wise unitaries to be used in ``feedforward``."""
        self.unitaries = unitaries

    def feedforward(
        self,
        node_states: Sequence[qt.Qobj],
    ) -> List[List[qt.Qobj]]:
        """
        node_states: list of qobj states, one per node
        Returns a list of layerwise states for each node.
        """
        if self.unitaries is None:
            raise RuntimeError("Unitaries not set; call random_network() or set_unitaries() first.")

        stored_states: List[List[qt.Qobj]] = []
        for state in node_states:
            layerwise = [state]
            current_state = state
            for layer in range(1, len(self.arch)):
                # local quantum convolution before the layer
                _ = self.conv.run(state.to_array().reshape(self.arch[layer - 1], self.arch[layer - 1]).astype(float))
                current_state = _layer_channel(self.arch, self.unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    def random_network(self, samples: int = 10):
        """Generate a random network and store the unitaries."""
        arch, unitaries, training_data, target_unitary = random_network(self.arch, samples)
        self.unitaries = unitaries
        return arch, unitaries, training_data, target_unitary

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)


__all__ = [
    "GraphQNNQML",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
