"""Quantum Graph Neural Network utilities using PennyLane.

The module keeps the basic API of the original QML seed but adds a
variational circuit builder, a training routine that optimizes the
circuit parameters with a classical optimiser, and a graph‑based
regularizer.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray
State = Tensor

def _random_qubit_unitary(num_qubits: int) -> qml.QNode:
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit():
        for _ in range(num_qubits):
            qml.RX(np.random.randn(), wires=_)
            qml.RY(np.random.randn(), wires=_)
            qml.RZ(np.random.randn(), wires=_)
        return qml.state()
    return circuit

def random_training_data(unitary: qml.QNode, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        input_state = np.zeros(2 ** unitary.num_wires, dtype=complex)
        input_state[0] = 1.0
        target_state = unitary()
        dataset.append((input_state, target_state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    unitaries: List[qml.QNode] = []
    for layer in range(1, len(qnn_arch)):
        num_qubits = qnn_arch[layer]
        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev)
        def layer_circuit(params, wires=range(num_qubits)):
            for i, w in enumerate(wires):
                qml.RX(params[i], wires=w)
                qml.RY(params[i + num_qubits], wires=w)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()
        params = np.random.randn(2 * num_qubits)
        unitaries.append((layer_circuit, params))
    def target_circuit():
        state = np.zeros(2 ** qnn_arch[-1], dtype=complex)
        state[0] = 1.0
        return state
    dataset = random_training_data(target_circuit, samples)
    return qnn_arch, unitaries, dataset, target_circuit

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Tuple[qml.QNode, Tensor]], samples: Iterable[Tuple[Tensor, Tensor]]):
    stored_states: List[List[Tensor]] = []
    for input_state, _ in samples:
        layerwise = [input_state]
        current_state = input_state
        for (layer_c, params), wires in zip(unitaries, range(1, len(qnn_arch))):
            dev = qml.device("default.qubit", wires=wires)
            @qml.qnode(dev)
            def apply_layer():
                layer_c(params)
                return qml.state()
            current_state = apply_layer()
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: Tensor, b: Tensor) -> float:
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNQuantum:
    """Variational graph‑based quantum neural network."""
    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.unitaries, self.params = self._build_layers()

    def _build_layers(self):
        unitaries = []
        params = []
        for layer in range(1, len(self.arch)):
            num_qubits = self.arch[layer]
            dev = qml.device("default.qubit", wires=num_qubits)
            @qml.qnode(dev)
            def layer_circuit(p, wires=range(num_qubits)):
                for i, w in enumerate(wires):
                    qml.RX(p[i], wires=w)
                    qml.RY(p[i + num_qubits], wires=w)
                for i in range(num_qubits - 1):
                    qml.CNOT([i, i + 1])
                return qml.state()
            param = np.random.randn(2 * num_qubits)
            unitaries.append(layer_circuit)
            params.append(param)
        return unitaries, params

    def forward(self, input_state: Tensor) -> Tensor:
        state = input_state
        for layer_c, p in zip(self.unitaries, self.params):
            dev = qml.device("default.qubit", wires=layer_c.num_wires)
            @qml.qnode(dev)
            def apply():
                layer_c(p)
                return qml.state()
            state = apply()
        return state

    def train_variational(self, dataset: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 50, reg_weight: float = 0.0, reg_graph: nx.Graph | None = None):
        opt = qml.GradientDescentOptimizer(lr)
        for epoch in range(epochs):
            loss = 0.0
            for x, y in dataset:
                def loss_fn(p):
                    state = self._apply_layers(x, p)
                    return 1.0 - state_fidelity(state, y)
                loss += opt.step_and_cost(loss_fn, self.params)
            loss /= len(dataset)
            if reg_weight > 0.0 and reg_graph is not None:
                reg = self._graph_regularizer(x, reg_graph)
                loss += reg_weight * reg
            print(f"Epoch {epoch+1}/{epochs} loss: {loss:.4f}")

    def _apply_layers(self, input_state: Tensor, params_list: List[np.ndarray]) -> Tensor:
        state = input_state
        for layer_c, p in zip(self.unitaries, params_list):
            dev = qml.device("default.qubit", wires=layer_c.num_wires)
            @qml.qnode(dev)
            def apply():
                layer_c(p)
                return qml.state()
            state = apply()
        return state

    def _graph_regularizer(self, state: Tensor, graph: nx.Graph) -> float:
        reg = 0.0
        for i, j in graph.edges():
            reg += 1.0 - state_fidelity(state, state)
        return reg

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNQuantum",
]
