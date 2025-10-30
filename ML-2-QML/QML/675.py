"""Quantum graph neural network with variational circuit and training loop.

This module implements a GraphQNN class that mirrors the classical
interface but uses a PennyLane variational circuit.  The network
consists of a sequence of layers, each applying a parameterised
single‑qubit rotation followed by a CNOT chain.  A training routine
optimises the parameters to approximate a target unitary that is
generated randomly for the final layer.

Only PennyLane, NumPy and NetworkX are required.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import numpy as np

Array = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> Array:
    """Return a random unitary matrix of shape (2**num_qubits, 2**num_qubits)."""
    dim = 2 ** num_qubits
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    U, _ = np.linalg.qr(A)
    return U


def random_training_data(unitary: Array, samples: int) -> List[Tuple[Array, Array]]:
    """Generate (input_state, target_state) pairs where target = U * input."""
    dataset: List[Tuple[Array, Array]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Array], List[Tuple[Array, Array]], Array]:
    """Create a random variational circuit and training data."""
    num_qubits = qnn_arch[-1]
    num_layers = len(qnn_arch) - 1
    # Parameters: list of arrays, one per layer, shape (num_qubits, 3)
    params: List[Array] = [np.random.randn(num_qubits, 3) for _ in range(num_layers)]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), params, training_data, target_unitary


def _rotate_layer(params: Array, wires: Sequence[int]) -> None:
    """Apply a single‑qubit rotation layer with parameters shape (num_qubits, 3)."""
    for w, (rx, ry, rz) in zip(wires, params):
        qml.Rot(rx, ry, rz, wires=w)


def _entangle_layer(wires: Sequence[int]) -> None:
    """Apply a nearest‑neighbour CNOT chain."""
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[Array],
    samples: Iterable[Tuple[Array, Array]],
) -> List[List[Array]]:
    """Propagate each sample through the variational circuit and collect state vectors."""
    num_qubits = qnn_arch[-1]
    dev = qml.device("default.qubit", wires=num_qubits)

    def circuit(state: Array, param_set: Array) -> Array:
        qml.StatePrep(state, wires=range(num_qubits))
        for layer_params in param_set:
            _rotate_layer(layer_params, range(num_qubits))
            _entangle_layer(range(num_qubits))
        return qml.state()

    qnode = qml.QNode(circuit, dev, interface="autograd")
    stored = []
    for inp, _ in samples:
        layerwise = [inp]
        for p in params:
            state = qnode(inp, [p])
            layerwise.append(state)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Array, b: Array) -> float:
    """Return the absolute squared overlap between pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Array],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN:
    """Quantum graph neural network with variational parameters and training support."""

    def __init__(self, architecture: Sequence[int]):
        self.architecture = list(architecture)
        self.num_qubits = self.architecture[-1]
        self.num_layers = len(self.architecture) - 1
        self.params: List[Array] = [
            np.random.randn(self.num_qubits, 3) for _ in range(self.num_layers)
        ]
        self.target_unitary: Array | None = None
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def set_target(self, unitary: Array) -> None:
        self.target_unitary = unitary

    def _circuit(self, state: Array, param_set: Sequence[Array]) -> Array:
        qml.StatePrep(state, wires=range(self.num_qubits))
        for layer_params in param_set:
            _rotate_layer(layer_params, range(self.num_qubits))
            _entangle_layer(range(self.num_qubits))
        return qml.state()

    def forward(self, state: Array) -> Array:
        qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        return qnode(state, self.params)

    def train(
        self,
        data: Iterable[Tuple[Array, Array]],
        epochs: int = 200,
        lr: float = 0.01,
    ) -> None:
        if self.target_unitary is None:
            raise ValueError("Target unitary not set. Use `set_target` before training.")
        opt = qml.AdamOptimizer(stepsize=lr)
        for _ in range(epochs):
            for inp, tgt in data:
                def cost_fn(params):
                    qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
                    out = qnode(inp, params)
                    return 1 - state_fidelity(out, tgt)
                self.params = opt.step(cost_fn, self.params)

    def graph_loss(
        self,
        samples: Iterable[Tuple[Array, Array]],
        threshold: float,
    ) -> float:
        """Compute a graph‑based loss over a batch of outputs."""
        activations = feedforward(self.architecture, self.params, samples)
        final_states = [acts[-1] for acts in activations]
        G = fidelity_adjacency(final_states, threshold)
        if G.number_of_edges() == 0:
            return 0.0
        avg_weight = sum(data["weight"] for _, _, data in G.edges(data=True)) / G.number_of_edges()
        return 1.0 - avg_weight

    def __repr__(self) -> str:
        return f"<GraphQNN arch={self.architecture} layers={self.num_layers}>"

__all__ = [
    "GraphQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
