"""GraphQNN - quantum module using Pennylane variational circuits.

This module mirrors the classical interface but implements state‑vector
propagation and a parameter‑driven variational Ansatz.  The graph‑based
utilities are kept identical to the classical version for compatibility in
hybrid workflows.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import numpy as np

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input_state, target_state) pairs where target = U * input."""
    dataset = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        inp = _random_qubit_state(num_qubits)
        tgt = unitary @ inp
        dataset.append((inp, tgt))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[np.ndarray], List[Tuple[Tensor, Tensor]], np.ndarray]:
    """Return architecture, a list of randomly initialized parameter sets per layer,
    training data and a target unitary.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    init_params = [np.random.randn(num) for num in qnn_arch[:-1]]
    return list(qnn_arch), init_params, training_data, target_unitary


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = np.abs(np.vdot(state_i, state_j)) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return np.abs(np.vdot(a, b)) ** 2


class GraphLoss:
    """Graph‑based loss for quantum states.  It mirrors :class:`ml_code.GraphLoss`
    but operates on NumPy arrays.  The loss is a weighted sum of squared
    differences between the predicted and target state amplitudes.
    """
    def __init__(self, adjacency: nx.Graph):
        self.adj_matrix = np.zeros((adjacency.number_of_nodes(),
                                    adjacency.number_of_nodes()), dtype=float)
        for i, j, data in adjacency.edges(data=True):
            w = data.get("weight", 1.0)
            self.adj_matrix[i, j] = w
            self.adj_matrix[j, i] = w

    def __call__(self, preds: np.ndarray, targets: np.ndarray) -> float:
        diff = preds - targets
        return float(np.sum((diff @ self.adj_matrix) * diff) /
                     (2 * self.adj_matrix.sum() + 1e-12))


class GraphQNN:
    """Quantum graph‑neural‑network that builds a variational Ansatz per layer.

    The network is represented as a list of parameter arrays, one per layer,
    and a Pennylane quantum device that executes the circuit.  The public API
    matches the classical version so that the two can be swapped in a hybrid
    training loop.
    """
    def __init__(self, qnn_arch: Sequence[int], device: str | qml.Device = None):
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[-1]
        self.dev = device or qml.device("default.qubit", wires=self.num_qubits)
        # Parameter lists per layer
        self.params: List[np.ndarray] = [np.random.randn(num) for num in self.arch[:-1]]
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            qml.QubitStateVector(inputs, wires=range(self.num_qubits))
            for layer_idx, layer_params in enumerate(params):
                for q in range(self.arch[layer_idx]):
                    qml.RX(layer_params[q], wires=q)
                for q in range(self.arch[layer_idx] - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()
        self.circuit = circuit

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return a list of state‑vector lists, one per sample."""
        stored: List[List[Tensor]] = []
        for inp, _ in samples:
            state = self.circuit(inp, self.params)
            stored.append([state])
        return stored

    def train_step(
        self,
        data_loader: Iterable[Tuple[Tensor, Tensor]],
        optimizer: qml.GradientDescentOptimizer,
        loss_fn: callable,
    ) -> float:
        total_loss = 0.0
        for inp, tgt in data_loader:
            def cost(p):
                pred = self.circuit(inp, p)
                return loss_fn(pred, tgt)
            loss, grad = optimizer.step_and_cost(cost, self.params)
            self.params = grad
            total_loss += loss
        return total_loss / len(data_loader)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
        return random_training_data(unitary, samples)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)


__all__ = [
    "GraphQNN",
    "GraphLoss",
    "fidelity_adjacency",
    "state_fidelity",
    "random_network",
    "random_training_data",
]
