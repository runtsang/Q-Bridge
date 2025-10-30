import itertools
from typing import Iterable, Sequence, Tuple, List

import pennylane as qml
import pennylane.numpy as np
import networkx as nx
from pennylane import utils

class GraphQNN:
    """Hybrid quantum‑classical graph neural network.

    The class builds a variational circuit based on the supplied architecture,
    provides a QNode for state preparation and measurement, and exposes
    fidelity‑based graph utilities for quantum states.
    """

    def __init__(self, qnn_arch: Sequence[int], dev: qml.Device | None = None):
        self.qnn_arch = list(qnn_arch)
        self.num_wires = max(self.qnn_arch)
        self.dev = dev or qml.device("default.qubit", wires=self.num_wires)
        self.unitaries: List[np.ndarray] = []
        self._random_network()
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _random_network(self) -> None:
        """Generate random unitary matrices for each layer."""
        self.unitaries = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            dim = 2 ** (in_f + out_f)
            self.unitaries.append(utils.random_unitary(dim))

    def _circuit(self, input_state: np.ndarray):
        """Quantum circuit that prepares the input state and applies the variational layers."""
        qml.StatePrep(input_state, wires=range(self.qnn_arch[0]))
        for layer, unitary in enumerate(self.unitaries):
            wires = range(self.qnn_arch[layer] + self.qnn_arch[layer + 1])
            qml.QubitUnitary(unitary, wires=wires)
        return qml.state()

    def feedforward(self, input_state: np.ndarray) -> np.ndarray:
        """Return the output state vector for the given input state."""
        return self.qnode(input_state)

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the absolute squared overlap between two pure states."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def random_training_data(self, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate input and target state pairs by propagating random states through the circuit."""
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            dim_in = 2 ** self.qnn_arch[0]
            state = np.random.randn(dim_in) + 1j * np.random.randn(dim_in)
            state /= np.linalg.norm(state)
            target = self.feedforward(state)
            dataset.append((state, target))
        return dataset

__all__ = ["GraphQNN"]
