"""
Hybrid Graph Quantum Neural Network – Quantum implementation.

The module mirrors the classical counterpart but operates on qutip.Qobj objects.
New utilities:
* unitary_distance_graph – builds a graph based on Hilbert–Schmidt distance of unitaries.
* run_full – convenience wrapper that generates a random network, propagates
  the training data through the layers and returns the final unitary graph.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc

Tensor = qt.Qobj


class GraphQNN__gen167:
    """
    Quantum graph neural network with extended utilities.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = arch

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = GraphQNN__gen167._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(
        qnn_arch: List[int], samples: int
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """
        Construct a random variational circuit.  Each layer contains a list of unitary
        gates that act on the concatenated input and a fresh ancilla space.
        """
        target_unitary = GraphQNN__gen167._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNN__gen167.random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = GraphQNN__gen167._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(
                        GraphQNN__gen167._random_qubit_unitary(num_inputs + 1),
                        GraphQNN__gen167._tensored_id(num_outputs - 1),
                    )
                    op = GraphQNN__gen167._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return GraphQNN__gen167._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, GraphQNN__gen167._tensored_zero(num_outputs))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return GraphQNN__gen167._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = GraphQNN__gen167._layer_channel(
                    qnn_arch, unitaries, layer, current_state
                )
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the absolute squared overlap between pure states a and b."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
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
            fid = GraphQNN__gen167.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def unitary_distance_graph(
        unitaries: Sequence[qt.Qobj],
        threshold: float = 0.1,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a graph where nodes correspond to unitaries.  Edge weights are the
        Hilbert–Schmidt distance:
        d(U,V) = sqrt(1 - |Tr(U† V)| / d).
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(unitaries)))

        def hs_distance(U: qt.Qobj, V: qt.Qobj) -> float:
            d = U.shape[0]
            tr = (U.dag() * V).tr()
            return math.sqrt(max(0.0, 1 - abs(tr) / d))

        for (i, U), (j, V) in itertools.combinations(enumerate(unitaries), 2):
            dist = hs_distance(U, V)
            if dist >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and dist >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def run_full(
        arch: Sequence[int],
        samples: int,
        *,
        threshold: float = 0.8,
    ) -> Tuple[List[List[qt.Qobj]], nx.Graph]:
        """
        Convenience wrapper that generates a random network, propagates the
        training data through the layers and returns the list of final states
        together with a fidelity adjacency graph.
        """
        arch, unitaries, training_data, _ = GraphQNN__gen167.random_network(arch, samples)

        final_states: List[qt.Qobj] = []
        for state, _ in training_data:
            current = state
            for layer in range(1, len(arch)):
                current = GraphQNN__gen167._layer_channel(arch, unitaries, layer, current)
            final_states.append(current)

        adjacency = GraphQNN__gen167.fidelity_adjacency(final_states, threshold=threshold)
        return unitaries, adjacency


__all__ = ["GraphQNN__gen167"]
