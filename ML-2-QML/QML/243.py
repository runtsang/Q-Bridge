"""Quantum graph neural network utilities using PennyLane.

The module reproduces the classical interface but replaces the linear layers
with parameter‑shaped unitary operations.  A variational circuit is
constructed for each layer and trained with the Adam optimiser on a
state‑vector simulator.  The fidelity adjacency graph is built from the
output states of a target unitary, allowing a direct comparison with the
classical counterpart.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np  # use pennylane's numpy for autodiff


class GraphQNN:
    """A simple quantum graph neural network wrapper."""
    def __init__(self, arch: Sequence[int], dev_name: str = "default.qubit", wires: int | None = None):
        self.arch = list(arch)
        self.dev_name = dev_name
        self.wires = wires or max(arch)
        self.dev = qml.device(dev_name, wires=self.wires)
        self.params = self._init_params()

    def _init_params(self) -> List[np.ndarray]:
        """Return a list of parameter arrays for each layer."""
        params = []
        for in_w, out_w in zip(self.arch[:-1], self.arch[1:]):
            shape = (in_w + out_w, 3)  # simple RY, RZ, RX per qubit
            params.append(np.random.uniform(0, 2 * np.pi, shape))
        return params

    # --------------------------------------------------------------------- #
    # Random data generation (unchanged from the seed)
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_training_data(target_unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset = []
        num_qubits = int(np.log2(target_unitary.shape[0]))
        for _ in range(samples):
            state = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
            state /= np.linalg.norm(state)
            dataset.append((state, target_unitary @ state))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple["GraphQNN", List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Return a GraphQNN instance, its parameters, training data and the target unitary."""
        dim = 2 ** arch[-1]
        random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        target_unitary = np.linalg.qr(random_matrix)[0]
        dataset = GraphQNN.random_training_data(target_unitary, samples)
        instance = GraphQNN(arch)
        instance.params = instance._init_params()
        return instance, instance.params, dataset, target_unitary

    # --------------------------------------------------------------------- #
    # Variational circuit for a single layer
    # --------------------------------------------------------------------- #
    def _layer_circuit(self, layer: int, state_in: np.ndarray) -> np.ndarray:
        in_w = self.arch[layer - 1]
        out_w = self.arch[layer]
        total_w = in_w + out_w
        @qml.qnode(self.dev, interface="autograd")
        def circuit(state):
            qml.StatePrep(state, wires=range(total_w))
            for i in range(total_w):
                qml.RX(self.params[layer][i, 0], wires=i)
                qml.RY(self.params[layer][i, 1], wires=i)
                qml.RZ(self.params[layer][i, 2], wires=i)
            return qml.state()
        padded = np.zeros(2 ** total_w, dtype=complex)
        padded[: len(state_in)] = state_in
        return circuit(padded)

    # --------------------------------------------------------------------- #
    # Forward propagation
    # --------------------------------------------------------------------- #
    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        stored: List[List[np.ndarray]] = []
        for state_in, _ in samples:
            layerwise = [state_in]
            current = state_in
            for layer in range(1, len(self.arch)):
                current = self._layer_circuit(layer, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    # --------------------------------------------------------------------- #
    # Fidelity utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        return abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Variational training
    # --------------------------------------------------------------------- #
    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 10, lr: float = 0.01) -> None:
        opt = qml.AdamOptimizer(lr)
        params = [p.copy() for p in self.params]
        for _ in range(epochs):
            for state_in, target in dataset:
                def cost(p):
                    self.params = p
                    out = state_in
                    for layer in range(1, len(self.arch)):
                        out = self._layer_circuit(layer, out)
                    return np.sum(np.abs(out - target) ** 2)
                params = opt.step(cost, params)
        self.params = params

    def evaluate(self, dataset: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        total = 0.0
        for state_in, target in dataset:
            out = state_in
            for layer in range(1, len(self.arch)):
                out = self._layer_circuit(layer, out)
            total += np.sum(np.abs(out - target) ** 2)
        return total / len(dataset)


__all__ = [
    "GraphQNN",
]
