"""Hybrid Graph Neural Network performing quantum state propagation with optional quanvolution front‑end.

The class `GraphQNNHybrid` mirrors the classical API but operates on qutip.Qobj states
and uses a quantum convolution filter (QuanvCircuit) to preprocess the input data.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt

# Import the quantum Conv filter from Conv.py seed
try:
    from Conv import Conv  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback if Conv cannot be imported
    def Conv():
        class DummyQuanv:
            def run(self, data):
                return qt.Qobj()
        return DummyQuanv()


def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return state.ptrace(keep)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random Haar‑distributed unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = qt.rand_unitary(dim)
    qobj = qt.Qobj(matrix)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Generate a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    ket = qt.rand_ket(dim)
    return ket


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a chain of random unitary layers and a training set."""
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

    return list(qnn_arch), unitaries, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj
) -> qt.Qobj:
    """Apply the unitary channel for a single layer."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_id(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
) -> List[List[qt.Qobj]]:
    """Run the quantum network on a batch of samples, storing intermediate states."""
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
    """Squared overlap between two pure states."""
    return float(abs((a.dag() * b)[0, 0]) ** 2)


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
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


class GraphQNNHybrid:
    """Hybrid quantum graph neural network with optional quanvolution front‑end."""

    def __init__(self, arch: Sequence[int], quanv_kernel: int = 2, threshold: float = 0.0):
        self.arch = list(arch)
        self.quanv = Conv()  # returns a QuanvCircuit instance
        # Adjust kernel/threshold if possible
        if hasattr(self.quanv, "n_qubits"):
            self.quanv.n_qubits = quanv_kernel ** 2
        if hasattr(self.quanv, "threshold"):
            self.quanv.threshold = threshold
        self.weights: List[List[qt.Qobj]] | None = None
        self.training_data: List[Tuple[qt.Qobj, qt.Qobj]] | None = None
        self.target_unitary: qt.Qobj | None = None

    def build_random(self, samples: int) -> None:
        """Generate random unitaries and synthetic training data."""
        _, self.weights, self.training_data, self.target_unitary = random_network(
            self.arch, samples
        )

    def run(self, data: qt.Qobj) -> List[qt.Qobj]:
        """Run the quanvolution filter followed by quantum channel propagation."""
        if self.weights is None:
            raise RuntimeError("Model has not been built; call `build_random` first.")
        # Apply quanvolution filter to the input state
        # The filter expects a 2‑D array; reshape the state vector accordingly.
        raw = data.full().reshape(self.quanv.n_qubits, 1) if hasattr(self.quanv, "n_qubits") else data.full()
        conv_state = self.quanv.run(raw)
        # Forward through the quantum layers
        activations = [conv_state]
        current = conv_state
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.weights, layer, current)
            activations.append(current)
        return activations

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
