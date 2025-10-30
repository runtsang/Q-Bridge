import pennylane as qml
import pennylane.numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

class GraphQNNHybrid:
    """Quantum graph neural network with a variational circuit and fidelity‑based graph."""
    def __init__(self, qnn_arch: Sequence[int], dev: qml.Device | None = None):
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = qnn_arch[-1]
        self.dev = dev or qml.device("default.qubit", wires=self.num_qubits)
        # Parameters: one 3‑parameter rotation per qubit per layer
        self.params = np.random.randn(len(qnn_arch)-1, self.num_qubits, 3)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: np.ndarray, params: np.ndarray):
        # Encode classical input x as angle rotations on each qubit
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Variational layers
        for layer_params in params:
            for i in range(self.num_qubits):
                qml.Rot(*layer_params[i], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        # Output expectation value of PauliZ on first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: np.ndarray) -> float:
        return self.qnode(x, self.params)

    def train_epoch(self,
                    data_loader: Iterable[Tuple[np.ndarray, np.ndarray]],
                    epochs: int = 1,
                    lr: float = 0.01):
        """Run one epoch of gradient descent on the variational parameters."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            loss = 0.0
            for x, target in data_loader:
                cost = lambda p: (self.qnode(x, p) - target) ** 2
                self.params = opt.step(cost, self.params)
                loss += cost(self.params)
            print(f"Epoch {epoch+1} - Cost: {loss:.4f}")

    @staticmethod
    def random_training_data(unitary: qml.QNode, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate random input–target pairs using a fixed unitary."""
        dataset = []
        for _ in range(samples):
            state = np.random.randn(2**unitary.device.num_wires) + 1j*np.random.randn(2**unitary.device.num_wires)
            state /= np.linalg.norm(state)
            target = unitary(state)
            dataset.append((state, target))
        return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit and matching training data."""
    dev = qml.device("default.qubit", wires=qnn_arch[-1])

    def random_circuit(x: np.ndarray, params: np.ndarray):
        for i in range(qnn_arch[-1]):
            qml.Rot(*params[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    params = np.random.randn(qnn_arch[-1], 3)
    unitary = qml.QNode(random_circuit, dev)
    training_data = GraphQNNHybrid.random_training_data(unitary, samples)
    return list(qnn_arch), [params], training_data, unitary

def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[np.ndarray]],
                samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Propagate a batch of states through the variational layers."""
    stored = []
    for x, _ in samples:
        states = [x]
        current = x
        for layer_params in unitaries:
            def layer_circuit(x, params):
                for i in range(len(params)):
                    qml.Rot(*params[i], wires=i)
                return qml.expval(qml.PauliZ(0))
            qnode_layer = qml.QNode(layer_circuit, qml.device("default.qubit", wires=len(x)))
            current = qnode_layer(current, layer_params)
            states.append(current)
        stored.append(states)
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure states."""
    return np.abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def train_graph_qnn(model: GraphQNNHybrid,
                    data_loader: Iterable[Tuple[np.ndarray, np.ndarray]],
                    epochs: int,
                    lr: float = 0.01):
    """Unified training loop for the quantum graph‑neural‑network."""
    for epoch in range(epochs):
        loss = 0.0
        for x, target in data_loader:
            cost = lambda p: (model.qnode(x, p) - target) ** 2
            model.params = qml.GradientDescentOptimizer(stepsize=lr).step(cost, model.params)
            loss += cost(model.params)
        print(f"Epoch {epoch+1} - Cost: {loss:.4f}")

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "train_graph_qnn",
]
