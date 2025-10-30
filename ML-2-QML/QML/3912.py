"""Hybrid Graph Neural Network – Quantum implementation.

The class :class:`GraphQNNGen224` provides the same API as its classical
counterpart but operates on qutip objects and qiskit circuits.  The
`conv_filter` method returns a tunable quanvolution circuit that can
be run on any qiskit backend.  All utility functions are documented
inline for clarity.

Typical usage:

    >>> from GraphQNN__gen224 import GraphQNNGen224
    >>> gnn = GraphQNNGen224([4, 8, 4])
    >>> arch, unitaries, train, target = gnn.random_network(samples=10)
    >>> states = gnn.feedforward(arch, unitaries, train)
    >>> gnn.fidelity_adjacency([s[-1] for s in states], 0.9)

"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt

Tensor = qt.Qobj


def _tensored_id(num_qubits: int) -> Tensor:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> Tensor:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: Tensor, source: int, target: int) -> Tensor:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(
        size=(dim, dim)
    )
    unitary = np.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> Tensor:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(
        size=(dim, 1)
    )
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random quantum network.

    Returns architecture, list of layer unitaries, training data and the target unitary.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Tensor] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(
                    _random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1)
                )
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: Tensor, keep: Sequence[int]) -> Tensor:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: Tensor, remove: Sequence[int]) -> Tensor:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    layer: int,
    input_state: Tensor,
) -> Tensor:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate all samples through the quantum network."""
    stored_states: List[List[Tensor]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Tensor],
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


class GraphQNNGen224:
    """Container for a quantum graph neural network architecture.

    Parameters
    ----------
    qnn_arch
        Sequence of layer widths, e.g. ``[4, 8, 4]``.
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.qnn_arch = list(qnn_arch)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random quantum network."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[Tensor]],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate samples through the network."""
        return feedforward(qnn_arch, unitaries, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute state fidelity for two pure quantum states."""
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )

    @staticmethod
    def random_training_data(unitary: Tensor, samples: int):
        """Alias for compatibility with the classical version."""
        return random_training_data(unitary, samples)

    @staticmethod
    def conv_filter(kernel_size: int = 2, backend_name: str = "qasm_simulator"):
        """Return a tunable quanvolution circuit.

        The circuit is built on top of a qiskit backend and can be
        executed as ``circuit.run(data)``.  The default backend is the
        Aer QASM simulator.
        """
        from.Conv import Conv as QuantumConv

        return QuantumConv(kernel_size=kernel_size, backend=qiskit.Aer.get_backend(backend_name))

__all__ = [
    "GraphQNNGen224",
]
