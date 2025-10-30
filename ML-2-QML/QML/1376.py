"""
GraphQNN__gen335.qml
Quantum‑classical hybrid implementation using Pennylane.
Provides a variational circuit that mirrors the classical architecture
and utilities for generating random unitaries, training data, and
fidelity‑based graphs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training data for a quantum circuit."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state = state / np.linalg.norm(state)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit and a matching training set."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Parameter matrix for each layer: (num_qubits, 3) rotation angles
    params: List[np.ndarray] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        layer_params = np.random.randn(out_f, 3)
        params.append(layer_params)
    return list(qnn_arch), params, training_data, target_unitary

def _layer_circuit(num_qubits: int, params: np.ndarray):
    """Apply a layer of single‑qubit rotations followed by a CNOT chain."""
    for q in range(num_qubits):
        qml.RX(params[q, 0], wires=q)
        qml.RY(params[q, 1], wires=q)
        qml.RZ(params[q, 2], wires=q)
    for q in range(num_qubits - 1):
        qml.CNOT(wires=[q, q + 1])

def feedforward(qnn_arch: Sequence[int], params: Sequence[np.ndarray], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    """Execute the variational circuit on a batch of input states."""
    device = qml.device("default.qubit", wires=qnn_arch[-1])

    @qml.qnode(device)
    def circuit(state, layer_params):
        qml.StatePrep(state, wires=range(len(state)))
        for layer, p in zip(qnn_arch[1:], layer_params):
            _layer_circuit(layer, p)
        return qml.state()

    outputs: List[np.ndarray] = []
    for state, _ in samples:
        outputs.append(circuit(state, params))
    return outputs

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure quantum states."""
    return np.abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---------------------------------------------------------------------------

class GraphQNNQ:
    """
    Variational graph‑based quantum neural network.

    The network consists of a stack of rotation layers followed by
    CNOT chains.  Parameters are trained with a simple gradient‑based
    optimizer to reproduce a target unitary.
    """

    def __init__(self, arch: Sequence[int], seed: int | None = None):
        self.arch = list(arch)
        if seed is not None:
            np.random.seed(seed)
        self.params: List[np.ndarray] = [
            np.random.randn(out, 3) for in_f, out in zip(self.arch[:-1], self.arch[1:])
        ]

    def forward(self, state: np.ndarray) -> np.ndarray:
        device = qml.device("default.qubit", wires=self.arch[-1])

        @qml.qnode(device)
        def circuit(state):
            qml.StatePrep(state, wires=range(len(state)))
            for layer, p in zip(self.arch[1:], self.params):
                _layer_circuit(layer, p)
            return qml.state()

        return circuit(state)

    def train(self, target_unitary: np.ndarray, data: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.01, epochs: int = 100):
        """Gradient‑based training of the variational parameters."""
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for inp, target in data:
                def cost_fn(params):
                    # Update params
                    for i, p in enumerate(params):
                        self.params[i] = p
                    out = self.forward(inp)
                    return 1 - state_fidelity(out, target)
                grads = opt.compute_gradient(cost_fn, self.params)
                self.params = opt.apply_gradients(zip(grads, self.params))
        return self

    def embed(self, state: np.ndarray) -> np.ndarray:
        """Return the final state vector after all layers."""
        return self.forward(state)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNQ",
]
