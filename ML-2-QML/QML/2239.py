"""Quantum version of GraphQuanvolutionQNN using qutip.

This module mirrors the classical GraphQuanvolutionQNN but replaces
the classical convolutional filter with a quantum circuit and
performs state propagation over the graph.  It retains the
fidelity‑based adjacency construction from the original QML
seed and adds a simple quantum message‑passing mechanism.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# Utility functions (original QML)
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

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #

class QuantumQuanvolutionFilter:
    """Apply a random unitary to a single‑qubit or multi‑qubit state."""
    def __init__(self, num_qubits: int):
        self.unitary = _random_qubit_unitary(num_qubits)
    def forward(self, state: qt.Qobj) -> qt.Qobj:
        return self.unitary * state

# --------------------------------------------------------------------------- #
# Hybrid quantum graph‑neural‑network class
# --------------------------------------------------------------------------- #

class GraphQuanvolutionQNN:
    """
    Quantum version of the hybrid GNN.  Each node state is first
    transformed by a small random unitary (the quantum quanvolution
    filter).  States are then propagated across the graph using
    partial‑trace message passing, and finally a global unitary
    maps the hidden representation to the output space.
    """
    def __init__(
        self,
        graph: nx.Graph,
        in_qubits: int,
        hidden_qubits: int,
        out_qubits: int,
    ):
        self.graph = graph
        self.in_qubits = in_qubits
        self.hidden_qubits = hidden_qubits
        self.out_qubits = out_qubits

        # One filter per node
        self.filters = {node: QuantumQuanvolutionFilter(in_qubits) for node in graph.nodes}

        # Random unitaries for hidden and output layers
        self.hidden_unitary = _random_qubit_unitary(hidden_qubits)
        self.out_unitary = _random_qubit_unitary(out_qubits)

    def forward(self, states: List[qt.Qobj]) -> List[qt.Qobj]:
        """
        Parameters
        ----------
        states : List[qt.Qobj]
            Initial pure states for each node.

        Returns
        -------
        List[qt.Qobj]
            Updated states after one message‑passing step.
        """
        new_states: List[qt.Qobj] = []

        for node, state in zip(self.graph.nodes, states):
            # Apply the node‑specific quanvolution filter
            state = self.filters[node].forward(state)

            # Aggregate neighbour states via tensor product
            neighbor_states = [states[n] for n in self.graph.neighbors(node)]
            if neighbor_states:
                combined = qt.tensor(state, *neighbor_states)
                # Apply hidden unitary and partial trace to keep the same qubit count
                combined = self.hidden_unitary * combined * self.hidden_unitary.dag()
                combined = _partial_trace_remove(combined, range(self.in_qubits))
                state = combined

            # Final output unitary
            state = self.out_unitary * state * self.out_unitary.dag()
            new_states.append(state)

        return new_states

    # --------------------------------------------------------------------- #
    # Convenience helpers that expose the original QML API
    # --------------------------------------------------------------------- #
    def random_graph_network(self, samples: int):
        """Generate a random graph and associated random unitaries."""
        qnn_arch, unitaries, training_data, target_unitary = random_network(
            [self.in_qubits, self.hidden_qubits, self.out_qubits], samples
        )
        return qnn_arch, unitaries, training_data, target_unitary

    def compute_fidelity_graph(self, states: Sequence[qt.Qobj], threshold: float):
        """Return a graph where edges are weighted by fidelity."""
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQuanvolutionQNN",
    "QuantumQuanvolutionFilter",
    "feedforward",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
