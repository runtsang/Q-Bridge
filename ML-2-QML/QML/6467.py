import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

class GraphQNN:
    """
    Variational quantum neural network that follows the same API as the classical
    implementation.  The architecture is a list of integers describing the
    number of qubits processed at each layer.
    """

    def __init__(self, arch: Sequence[int], wires: int | None = None):
        self.arch = list(arch)
        self.wires = wires or arch[-1]
        self.n_layers = len(arch) - 1
        self.dev = qml.device("default.qubit", wires=self.wires)
        # one 3‑parameter rotation per qubit per layer (rx, ry, rz)
        self.params = np.random.randn(self.n_layers, self.wires, 3)

    def _circuit(self, x: np.ndarray, params: np.ndarray):
        """Variational circuit that prepares the output state for a given input."""
        # Encode classical input x as computational basis state
        for w, bit in enumerate(x):
            if bit:
                qml.PauliX(w)
        # Apply variational layers
        for l in range(self.n_layers):
            for q in range(self.wires):
                qml.RX(params[l, q, 0], wires=q)
                qml.RY(params[l, q, 1], wires=q)
                qml.RZ(params[l, q, 2], wires=q)
            # Entangling pattern – linear nearest‑neighbour CNOT chain
            for q in range(self.wires - 1):
                qml.CNOT(wires=[q, q + 1])
        # Return measurement of Z for each qubit
        return [qml.expval(qml.PauliZ(q)) for q in range(self.wires)]

    @property
    def qnode(self):
        return qml.QNode(self._circuit, self.dev)

    def feedforward(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[float]]:
        """Run all samples through the circuit and return outputs."""
        qnode = self.qnode
        outputs: List[List[float]] = []
        for x, _ in samples:
            out = qnode(x, self.params)
            outputs.append(out)
        return outputs

    def random_network(self, samples: int = 100) -> Tuple[List[int], np.ndarray, List[Tuple[np.ndarray, np.ndarray]], None]:
        """Create a random variational circuit and synthetic training data."""
        # Re‑randomise parameters
        self.params = np.random.randn(self.n_layers, self.wires, 3)
        qnode = self.qnode
        training_data: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            x = np.random.randint(0, 2, size=self.wires)
            y = qnode(x, self.params)
            training_data.append((x, np.array(y)))
        return self.arch, self.params, training_data, None

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from fidelity between pure states."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = np.abs(np.vdot(a, b)) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(
        self,
        training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 10,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        """Optimise the variational parameters with a simple MSE loss."""
        opt = qml.GradientDescentOptimizer(lr)
        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y_true in training_data:
                def cost_fn(p):
                    y_pred = self.qnode(x, p)
                    return np.mean((np.array(y_pred) - y_true) ** 2)

                self.params, loss = opt.step_and_cost(cost_fn, self.params)
                epoch_loss += loss
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")
        return losses
