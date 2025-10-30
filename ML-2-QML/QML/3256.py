"""FraudGraphHybrid: quantum side of the hybrid fraud‑detection model.

This module implements a graph‑based quantum neural network that
mirrors the photonic‑style classical network.  Each node of the graph
corresponds to a layer of the quantum circuit and each edge represents
a fidelity‑based adjacency that can be used for message‑passing or
graph‑convolution.  The module exposes the same training‑data
generation, feed‑forward, and fidelity utilities as the classical
side, allowing joint experiments.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List, Optional

import itertools
import networkx as nx
import qutip as qt
import scipy as sc


# --------------------------------------------------------------------------- #
# Utility functions for tensor identities and random unitaries
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Random training data
# --------------------------------------------------------------------------- #
def random_training_data(
    unitary: qt.Qobj,
    samples: int,
) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate a dataset of random pure states and their images under `unitary`."""
    num_qubits = len(unitary.dims[0])
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        transformed = unitary * state
        dataset.append((state, transformed))
    return dataset


# --------------------------------------------------------------------------- #
# Feed‑forward helper
# --------------------------------------------------------------------------- #
def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
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
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states a and b."""
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
# Quantum counterpart of the classical FraudGraphHybrid
# --------------------------------------------------------------------------- #
class FraudGraphHybrid:
    """Quantum counterpart of the classical FraudGraphHybrid.

    The class builds a layered unitary graph that mimics the photonic
    network.  It offers methods to generate random training data,
    perform a feed‑forward propagation that records intermediate states,
    and construct a fidelity‑based adjacency graph.  The API mirrors
    the classical side, enabling side‑by‑side experiments.
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.arch = list(qnn_arch)
        self.unitaries = self._build_random_network(self.arch)
        self.target_unitary = _random_qubit_unitary(self.arch[-1])

    @staticmethod
    def _build_random_network(qnn_arch: Sequence[int]) -> List[List[qt.Qobj]]:
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
        return unitaries

    def random_training_data(self, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate a dataset of random pure states and their images under the target unitary."""
        return random_training_data(self.target_unitary, samples)

    def feedforward(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        return feedforward(self.arch, self.unitaries, samples)

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "FraudGraphHybrid",
]
