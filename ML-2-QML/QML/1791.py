"""Quantum graph neural network implementation with Pennylane.

This module provides:
* A VariationalQNN class that builds a parameter‑sharded circuit.
* Random network generation that returns a list of parameterized circuits.
* Forward pass using a Pennylane QNode.
* Fidelity computation and adjacency construction.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of shape (2**n, 2**n)."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate state → U|state> pairs."""
    data = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        data.append((state, unitary @ state))
    return data

def random_network(qnn_arch: List[int], samples: int):
    """Return a list of parameter‑sharded unitary blocks."""
    unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(unitary, samples)

    layers: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for out in range(out_f):
            op = _random_qubit_unitary(in_f + 1)
            layer_ops.append(op)
        layers.append(layer_ops)
    return qnn_arch, layers, training_data, unitary

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared magnitude of inner product of two pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: List[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Variational quantum neural network
# --------------------------------------------------------------------------- #
class VariationalQNN:
    """Parameter‑sharded variational circuit.

    Each layer consists of single‑qubit rotations on all qubits.
    """

    def __init__(self, qnn_arch: List[int], device: str = "default.qubit"):
        self.arch = qnn_arch
        self.num_qubits = qnn_arch[-1]
        self.device = device

        self.dev = qml.device(device, wires=self.num_qubits)

        # Number of parameters: 3 per qubit per layer
        self.params_per_layer = 3 * self.num_qubits
        self.num_params = self.params_per_layer * (len(qnn_arch) - 1)

        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, *params):
        """Build a parameter‑sharded circuit."""
        param_idx = 0
        for _ in range(len(self.arch) - 1):
            for q in range(self.num_qubits):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                lam = params[param_idx + 2]
                qml.Rot(theta, phi, lam, wires=q)
                param_idx += 3
        return qml.state()

    def forward(self, init_state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply the variational circuit to an initial state."""
        self.dev.set_state(init_state)
        return self.qnode(*params)

    def get_num_params(self) -> int:
        return self.num_params

# --------------------------------------------------------------------------- #
# Forward pass helper
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: List[int],
    layers: List[List[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Simulate a forward pass of the parameter‑sharded circuit."""
    outputs: List[List[np.ndarray]] = []
    for init_state, _ in samples:
        state = init_state
        layerwise = [state]
        for layer_ops in layers:
            new_states = []
            for op in layer_ops:
                state = op @ state
                new_states.append(state)
            state = new_states[0]
            layerwise.append(state)
        outputs.append(layerwise)
    return outputs

__all__ = [
    "VariationalQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
