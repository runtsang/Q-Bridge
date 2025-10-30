"""Hybrid quantum graph neural network.

This module mirrors the classical GraphQNN but replaces linear layers
with variational unitaries.  Each layer is a unitary acting on the
current state plus an extra qubit; the output is obtained by tracing
out the input qubits.  Fidelity‑based adjacency graphs and a simple
training data generator are also provided.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import qutip as qt

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
    "feedforward",
    "build_graph",
]

class GraphQNN:
    """Quantum graph neural network based on variational unitaries.

    Parameters
    ----------
    *qnn_arch : int
        Sequence of layer sizes.  The first entry is the number of
        input qubits; each subsequent entry specifies the number of
        output qubits for that layer.
    """

    def __init__(self, *qnn_arch: int):
        self.arch = list(qnn_arch)

    def _layer_channel(
        self,
        layer: int,
        unitaries: Sequence[Sequence[qt.Qobj]],
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = self.arch[layer - 1]
        num_outputs = self.arch[layer]
        # prepend the input state to fresh zero state for the outputs
        state = qt.tensor(input_state, qt.fock(2 ** num_outputs).proj())
        U = unitaries[layer][0]
        for gate in unitaries[layer][1:]:
            U = gate * U
        new_state = U * state * U.dag()
        keep = list(range(num_outputs))
        return new_state.ptrace(keep)

    def feedforward(
        self,
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        stored: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise: List[qt.Qobj] = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, unitaries, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    def build_graph(
        self,
        states: List[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = qt.rand_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    target_unitary = qt.rand_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for _ in range(num_outputs):
            op = qt.rand_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b).full()[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    qnn = GraphQNN(*qnn_arch)
    return qnn.feedforward(unitaries, samples)
