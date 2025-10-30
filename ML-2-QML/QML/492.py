"""Quantum graph neural network with variational circuits and fidelity loss.

The class GraphQNN__gen130 mirrors the classical API but uses Pennylane to build a
variational circuit. The network encodes a classical input vector into a quantum
state, applies a stack of strongly entangling layers, and outputs the full
state vector. Training is performed by minimizing the fidelity loss between the
predicted state and a target state using Adam optimisation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

Tensor = np.ndarray


class GraphQNN__gen130:
    """Quantum graph neural network with variational layers."""

    def __init__(self, qnn_arch: Sequence[int]):
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[-1]
        self.num_layers = max(1, len(self.arch) - 1)

    # --------------------------------------------------------------------------- #
    # Core utilities (static for API compatibility)
    # --------------------------------------------------------------------------- #

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Tensor:
        dim = 2**num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def random_training_data(num_qubits: int, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate (input_vec, target_state) pairs."""
        data = []
        for _ in range(samples):
            # Classical input vector (real) of length num_qubits
            input_vec = np.random.randn(num_qubits)
            # Target quantum state of dimension 2**num_qubits
            target_state = GraphQNN__gen130._random_qubit_state(num_qubits)
            data.append((input_vec, target_state))
        return data

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], Tensor, List[Tuple[Tensor, Tensor]], List[Tensor]]:
        """Return architecture, initial parameters, training data and target states.

        Parameters are initialised for a stack of StronglyEntanglingLayers
        with one layer per hidden dimension in ``qnn_arch``.
        """
        num_qubits = qnn_arch[-1]
        num_layers = max(1, len(qnn_arch) - 1)
        init_params = np.random.randn(num_layers, num_qubits, 3)
        training_data = GraphQNN__gen130.random_training_data(num_qubits, samples)
        target_states = [t[1] for t in training_data]
        return list(qnn_arch), init_params, training_data, target_states

    @staticmethod
    def _circuit(num_qubits: int, num_layers: int):
        """Return a QNode that implements the variational circuit."""
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(x: np.ndarray, params: np.ndarray) -> Tensor:
            qml.AngleEmbedding(x, wires=range(num_qubits))
            for layer in range(num_layers):
                qml.StronglyEntanglingLayers(params[layer], wires=range(num_qubits))
            return qml.state()
        return circuit

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], params: Tensor, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        """Run the variational circuit on a batch of samples."""
        num_qubits = qnn_arch[-1]
        num_layers = max(1, len(qnn_arch) - 1)
        circuit = GraphQNN__gen130._circuit(num_qubits, num_layers)
        outputs = []
        for input_vec, _ in samples:
            outputs.append(circuit(input_vec, params))
        return outputs

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two pure states."""
        return float(np.abs(np.vdot(a, b))**2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen130.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------------- #

    @staticmethod
    def fidelity_loss(pred: Tensor, target: Tensor) -> float:
        """Loss defined as 1 - fidelity."""
        return 1.0 - GraphQNN__gen130.state_fidelity(pred, target)

    @staticmethod
    def train_one_epoch(params: Tensor, training_data: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 1) -> Tensor:
        """Perform a single epoch of Adam optimisation on the variational parameters.

        Parameters
        ----------
        params : np.ndarray
            Initial parameters of shape (num_layers, num_qubits, 3).
        training_data : list of (input_vec, target_state)
        lr : float
            Learning rate for Adam.
        epochs : int
            Number of full passes over the data.
        """
        num_qubits = params.shape[1]
        num_layers = params.shape[0]
        opt = qml.AdamOptimizer(stepsize=lr)
        circuit = GraphQNN__gen130._circuit(num_qubits, num_layers)

        for _ in range(epochs):
            for input_vec, target_state in training_data:
                def loss_fn(p):
                    pred = circuit(input_vec, p)
                    return GraphQNN__gen130.fidelity_loss(pred, target_state)

                params = opt.step(loss_fn, params)
        return params


# --------------------------------------------------------------------------- #
# Module‑level wrappers for API compatibility
# --------------------------------------------------------------------------- #

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], Tensor, List[Tuple[Tensor, Tensor]], List[Tensor]]:
    return GraphQNN__gen130.random_network(qnn_arch, samples)

def feedforward(qnn_arch: Sequence[int], params: Tensor, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tensor]:
    return GraphQNN__gen130.feedforward(qnn_arch, params, samples)

def state_fidelity(a: Tensor, b: Tensor) -> float:
    return GraphQNN__gen130.state_fidelity(a, b)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    return GraphQNN__gen130.fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

def train_one_epoch(params: Tensor, training_data: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 1) -> Tensor:
    return GraphQNN__gen130.train_one_epoch(params, training_data, lr=lr, epochs=epochs)

__all__ = [
    "GraphQNN__gen130",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "train_one_epoch",
]
