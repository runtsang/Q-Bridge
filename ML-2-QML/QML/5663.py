"""
GraphQNN – Quantum variational circuit with graph‑based regularisation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q


def random_training_data(
    target_unitary: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data by applying a target unitary to random states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        target = target_unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(arch: Sequence[int], samples: int):
    """Return architecture, dummy unitaries, training data and the target unitary."""
    dim = arch[-1]
    target_unitary = _random_qubit_unitary(dim)
    training_data = random_training_data(target_unitary, samples)
    # Dummy unitaries list – the GraphQNN class will generate its own params
    unitaries = None
    return list(arch), unitaries, training_data, target_unitary


class GraphQNN:
    """Quantum variational circuit with optional graph‑based regularisation."""

    def __init__(self, arch: Sequence[int], backend: str = "default.qubit"):
        self.arch = list(arch)
        self.wires = list(range(self.arch[-1]))
        self.device = qml.device(backend, wires=self.wires)
        self.params = self._init_params()
        self.optimizer = qml.AdamOptimizer(0.01)

    def _init_params(self) -> List[Tensor]:
        """Initialise a list of rotation parameters for each layer."""
        params: List[Tensor] = []
        for _ in range(1, len(self.arch)):
            layer_params = np.random.randn(len(self.wires), 3)
            params.append(layer_params)
        return params

    def _qnode(self, params: List[Tensor]):
        """Return a PennyLane QNode that applies the variational circuit."""
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            for layer_params in params:
                for i, (theta, phi, lam) in enumerate(layer_params):
                    qml.Rot(theta, phi, lam, wires=self.wires[i])
                # Entangle adjacent qubits
                for i in range(len(self.wires) - 1):
                    qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            return qml.state()
        return circuit()

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Compute the squared overlap between two pure state vectors."""
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)

    def build_graph_from_states(
        self,
        states: List[Tensor],
        threshold: float = 0.9,
    ) -> nx.Graph:
        """Create a graph where nodes are state vectors and edges are high‑fidelity pairs."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
        return graph

    def graph_loss(self, graph: nx.Graph) -> float:
        """Return a simple penalty proportional to the total edge weight."""
        if graph.number_of_edges() == 0:
            return 0.0
        return sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True))

    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 0.01,
        graph_reg: float = 0.0,
        graph_threshold: float = 0.9,
    ) -> None:
        self.optimizer = qml.AdamOptimizer(lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for state, target in data:
                def cost(params):
                    output = self._qnode(params)
                    loss = np.mean((output - target) ** 2)
                    if graph_reg > 0.0:
                        graph = self.build_graph_from_states([output], threshold=graph_threshold)
                        loss += graph_reg * self.graph_loss(graph)
                    return loss

                self.params = self.optimizer.step(cost, self.params)
                epoch_loss += cost(self.params)
            # Uncomment to see progress
            # print(f"Epoch {epoch}: loss={epoch_loss/len(data):.4f}")

    def evaluate(
        self, data: List[Tuple[Tensor, Tensor]]
    ) -> float:
        total_loss = 0.0
        for state, target in data:
            output = self._qnode(self.params)
            total_loss += np.mean((output - target) ** 2)
        return total_loss / len(data)


__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
]
