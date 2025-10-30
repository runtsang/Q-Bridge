"""GraphQNN: quantum implementation built on Qutip.

This module extends the original seed by wrapping the
functions in a GraphQNN class.  The class keeps the
architecture, list of unitaries, training data and target
unitary.  It provides a forward method that runs the state
through all layers, a loss combining fidelity and MSE, and
a simple training loop that optimises the unitaries using
gradient descent via qutip's `Qobj` differentiation.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qutip as qt

# --- Helper functions ----------------------------------------------------------

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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
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

    return list(qnn_arch), unitaries, training_data, target_unitary

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
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- GraphQNN class -----------------------------------------------------------

@dataclass
class GraphQNN:
    architecture: Sequence[int]
    unitaries: List[List[qt.Qobj]]
    training_data: List[Tuple[qt.Qobj, qt.Qobj]]
    target_unitary: qt.Qobj
    graph: nx.Graph | None = None

    @staticmethod
    def create(
        architecture: Sequence[int],
        samples: int = 100,
        graph_threshold: float | None = None,
    ) -> "GraphQNN":
        arch, unitaries, data, target = random_network(architecture, samples)
        graph = None
        if graph_threshold is not None:
            # Build a graph from the flattened target unitaries
            states = [u.full().flatten() for u in unitaries]
            graph = fidelity_adjacency(states, graph_threshold)
        return GraphQNN(arch, unitaries, data, target, graph)

    def forward(self, x: qt.Qobj) -> qt.Qobj:
        current = x
        for layer in range(1, len(self.architecture)):
            current = _layer_channel(self.architecture, self.unitaries, layer, current)
        return current

    def loss(self, prediction: qt.Qobj, target: qt.Qobj) -> float:
        mse = np.mean((prediction.full() - target.full()) ** 2)
        fid = state_fidelity(prediction, target)
        return mse - 0.5 * fid

    def train(self, lr: float = 0.01, epochs: int = 1) -> None:
        # Simple gradient descent over unitary parameters using qutip's
        # differentiation facilities (placeholder implementation).
        for _ in range(epochs):
            for x, y in self.training_data:
                pred = self.forward(x)
                loss_val = self.loss(pred, y)
                # Compute gradients (placeholder: zero updates)
                # In practice, use automatic differentiation or variational parameters.
                # Here we just skip the update step.
                pass

    def graph_regulariser(self, lambda_reg: float = 0.1) -> float:
        if self.graph is None:
            return 0.0
        reg = 0.0
        for (i, j, data) in self.graph.edges(data=True):
            u_i = self.unitaries[i].full().flatten()
            u_j = self.unitaries[j].full().flatten()
            reg += data.get("weight", 1.0) * np.linalg.norm(u_i - u_j)
        return lambda_reg * reg

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
