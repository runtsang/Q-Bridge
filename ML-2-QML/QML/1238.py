import pennylane as qml
import pennylane.numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

class GraphQNN:
    """
    Quantum graph neural network implemented with Pennylane.
    """
    def __init__(self, qnn_arch: Sequence[int], wires: int | None = None):
        self.arch = list(qnn_arch)
        self.wires = wires if wires is not None else max(self.arch)
        self.dev = qml.device("default.qubit", wires=self.wires)
        # Parameters for each layer: a list of arrays of rotation angles
        self.params: List[np.ndarray] = [
            np.random.randn(self.arch[layer]) for layer in range(1, len(self.arch))
        ]

    def _layer(self, layer: int, params: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Apply a variational layer defined by rotation angles.
        """
        @qml.qnode(self.dev, interface="autograd")
        def circuit(state_vec, angles):
            qml.StatePrep(state_vec, wires=range(self.arch[layer]))
            for i in range(self.arch[layer]):
                qml.RZ(angles[i], wires=i)
                qml.RX(angles[i], wires=i)
            for i in range(self.arch[layer] - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()
        return circuit(state, params)

    def feedforward(self, params: List[np.ndarray], x: np.ndarray) -> List[np.ndarray]:
        """
        Return the list of states after each layer.
        """
        states: List[np.ndarray] = [x]
        current = x
        for layer, layer_params in enumerate(params, start=1):
            current = self._layer(layer, layer_params, current)
            states.append(current)
        return states

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Squared overlap between two pure states.
        """
        return abs((a.conj().T @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph from state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_qubit_state(num_qubits: int) -> np.ndarray:
        """
        Generate a random pure state vector.
        """
        dim = 2 ** num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic dataset by applying a target unitary to random states.
        """
        data: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            state = GraphQNN.random_qubit_state(int(np.log2(unitary.shape[0])))
            target = unitary @ state
            data.append((state, target))
        return data

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Create a random network with a target unitary and synthetic dataset.
        """
        # Target unitary for the final layer
        dim = 2 ** qnn_arch[-1]
        random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(random_matrix)
        target_unitary = q
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        # Parameter lists for each layer
        params = [np.random.randn(qnn_arch[layer]) for layer in range(1, len(qnn_arch))]
        return list(qnn_arch), params, training_data, target_unitary

    def train(self, data: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.01, epochs: int = 200) -> None:
        """
        Train the quantum circuit end‑to‑end using Pennylane's autograd.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for state, target in data:
                def loss_fn(params):
                    out = self.feedforward(params, state)[-1]
                    return np.sum(np.abs(out - target) ** 2)
                self.params = opt.step(loss_fn, self.params)
