"""Quantum graph neural network with variational training."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qutip as qt

Qobj = qt.Qobj

@dataclass
class GraphQNNGen:
    """
    A quantum graph neural network that learns a target unitary.

    The network is a stack of layers, each represented by a unitary acting
    on the current register plus a fresh ancilla qubit.  For training we
    use a simple variational ansatz consisting of RY rotations on each
    qubit of every layer.  The loss is the mean‑squared difference of
    amplitudes between the target state and the variational state.
    An early‑stopping criterion monitors the fidelity‑based adjacency
    graph of the last‑layer states.
    """
    qnn_arch: Sequence[int]
    layers: List[List[Qobj]]
    target_unitary: Qobj
    training_data: List[Tuple[Qobj, Qobj]]
    fidelity_graph: nx.Graph | None = None

    @staticmethod
    def _random_unitary(num_qubits: int) -> Qobj:
        dim = 2 ** num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        mat, _ = np.linalg.qr(mat)
        return qt.Qobj(mat)

    @staticmethod
    def _zero_state(num_qubits: int) -> Qobj:
        zero = qt.basis(2, 0)
        state = zero
        for _ in range(num_qubits - 1):
            state = qt.tensor(state, zero)
        return state

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 100, seed: int | None = None) -> "GraphQNNGen":
        """Create a random quantum network and training data."""
        if seed is not None:
            np.random.seed(seed)
        layers: List[List[Qobj]] = [[]]
        for out in qnn_arch[1:]:
            layer_ops: List[Qobj] = []
            for _ in range(out):
                op = GraphQNNGen._random_unitary(out)
                layer_ops.append(op)
            layers.append(layer_ops)
        target_unitary = layers[-1][0]
        training_data: List[Tuple[Qobj, Qobj]] = []
        for _ in range(samples):
            state = GraphQNNGen._zero_state(qnn_arch[0])
            training_data.append((state, target_unitary * state))
        return GraphQNNGen(qnn_arch, layers, target_unitary, training_data)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
        """Return the state after each layer for each sample."""
        all_states: List[List[Qobj]] = []
        for state, _ in samples:
            states = [state]
            current = state
            for layer in range(1, len(self.qnn_arch)):
                op = self.layers[layer][0]
                current = op * current
                states.append(current)
            all_states.append(states)
        return all_states

    # ------------------------------------------------------------------ #
    # Fidelity utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Qobj, b: Qobj) -> float:
        """Squared overlap for pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def build_fidelity_graph(self,
                             threshold: float,
                             secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from the last‑layer states of the training data."""
        states = [states[-1] for states in self.feedforward(self.training_data)]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(sa, sb)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        self.fidelity_graph = graph
        return graph

    # ------------------------------------------------------------------ #
    # Variational training
    # ------------------------------------------------------------------ #
    def _compute_loss(self, theta: np.ndarray) -> float:
        """Compute mean‑squared loss for given rotation angles."""
        base_state = self._zero_state(self.qnn_arch[0])
        var_unitary = qt.qeye(2 ** self.qnn_arch[-1])
        for l in range(len(self.qnn_arch) - 1):
            for q in range(self.qnn_arch[l + 1]):
                angle = theta[l, q]
                ry = qt.Qobj([[np.cos(angle / 2), -np.sin(angle / 2)],
                              [np.sin(angle / 2), np.cos(angle / 2)]])
                ops = [qt.qeye(2)] * self.qnn_arch[l + 1]
                ops[q] = ry
                rot = qt.tensor(*ops)
                var_unitary = rot * var_unitary
        var_state = var_unitary * base_state
        target_state = self.target_unitary * base_state
        diff = var_state.full() - target_state.full()
        loss = np.mean(np.abs(diff) ** 2)
        return loss

    def train_variational(self,
                          epochs: int = 100,
                          lr: float = 0.1,
                          eps: float = 1e-3,
                          fidelity_threshold: float = 0.99,
                          patience: int = 5) -> None:
        """Variational optimisation of a simple RY‑rotation per qubit."""
        theta = np.random.randn(len(self.qnn_arch) - 1,
                                self.qnn_arch[-1])
        best_graph = None
        no_improve = 0
        for epoch in range(1, epochs + 1):
            loss = self._compute_loss(theta)
            grads = np.zeros_like(theta)
            for l in range(theta.shape[0]):
                for q in range(theta.shape[1]):
                    theta_plus = theta.copy()
                    theta_minus = theta.copy()
                    theta_plus[l, q] += eps
                    theta_minus[l, q] -= eps
                    loss_plus = self._compute_loss(theta_plus)
                    loss_minus = self._compute_loss(theta_minus)
                    grads[l, q] = (loss_plus - loss_minus) / (2 * eps)
            theta -= lr * grads

            if epoch % 5 == 0:
                graph = self.build_fidelity_graph(fidelity_threshold)
                if best_graph is not None and nx.is_isomorphic(graph, best_graph):
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    best_graph = graph
                    no_improve = 0

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def evaluate(self, state: Qobj) -> Qobj:
        """Apply the network to a single state."""
        current = state
        for layer in range(1, len(self.qnn_arch)):
            op = self.layers[layer][0]
            current = op * current
        return current

    def __repr__(self) -> str:
        return f"<GraphQNNGen arch={self.qnn_arch} samples={len(self.training_data)}>"

__all__ = ["GraphQNNGen"]
