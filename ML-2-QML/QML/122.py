import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np

Tensor = np.ndarray

def _random_unitary_params(num_qubits: int) -> np.ndarray:
    """Return a random parameter array for a StronglyEntanglingLayers gate."""
    return np.random.normal(size=(3, num_qubits))

def random_training_data(unitary_params: List[np.ndarray],
                         samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a unitary target."""
    data = []
    total_qubits = sum(p.shape[1] for p in unitary_params)
    dim = 2 ** total_qubits
    for _ in range(samples):
        # random input state
        x = np.random.normal(size=dim)
        x = x / np.linalg.norm(x)
        # target state by applying the target unitary
        dev = qml.device("default.qubit", wires=total_qubits)
        @qml.qnode(dev, interface="autograd")
        def target_circuit(inp):
            qml.StatePrep(inp, wires=range(total_qubits))
            wire_offset = 0
            for params in unitary_params:
                wires = range(wire_offset, wire_offset + params.shape[1])
                qml.templates.StronglyEntanglingLayers(params, wires=wires)
                wire_offset += params.shape[1]
            return qml.state()
        y = target_circuit(x)
        data.append((x, y))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network."""
    params = [_random_unitary_params(n) for n in qnn_arch]
    training_data = random_training_data(params, samples)
    target_params = params  # For simplicity, the target is the same as the network
    return list(qnn_arch), params, training_data, target_params

class GraphQNN:
    """Quantum graph neural network using Pennylane.

    The network is a stack of StronglyEntanglingLayers applied to the input
    state.  Dropout is emulated by randomly applying Pauli‑Z gates to a subset
    of output qubits after each layer.
    """

    def __init__(self, qnn_arch: Sequence[int],
                 dev: qml.Device | None = None,
                 dropout: float = 0.0):
        self.arch = list(qnn_arch)
        self.dropout = dropout
        total_qubits = sum(self.arch)
        self.dev = dev or qml.device("default.qubit", wires=total_qubits)
        self.params = self._random_params()
        self.qnode = self._build_qnode()

    def _random_params(self) -> List[np.ndarray]:
        return [_random_unitary_params(n) for n in self.arch]

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inp, params):
            qml.StatePrep(inp, wires=range(self.dev.num_wires))
            wire_offset = 0
            for layer_params in params:
                n_q = layer_params.shape[1]
                wires = range(wire_offset, wire_offset + n_q)
                qml.templates.StronglyEntanglingLayers(layer_params, wires=wires)
                if self.dropout > 0.0:
                    for w in wires:
                        if np.random.rand() < self.dropout:
                            qml.PauliZ(w)
                wire_offset += n_q
            return qml.state()
        return circuit

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run the quantum circuit on a batch of samples."""
        all_states: List[List[Tensor]] = []
        for inp, _ in samples:
            state = self.qnode(inp, self.params)
            all_states.append([state])
        return all_states

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the absolute squared overlap between two pure states."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(self, dataset: Iterable[Tuple[Tensor, Tensor]],
              lr: float = 0.01, epochs: int = 10) -> None:
        """Train the quantum circuit to map inputs to target states."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for inp, target in dataset:
                def loss_fn(params):
                    out = self.qnode(inp, params)
                    return np.mean((out - target) ** 2)
                self.params = opt.step(loss_fn, self.params)

    @staticmethod
    def compare_models(a: "GraphQNN", b: "GraphQNN",
                       threshold: float = 0.8,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Compare the parameters of two quantum models."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(a.params)))
        for (i, pa), (j, pb) in itertools.combinations(enumerate(a.params), 2):
            fid = GraphQNN.state_fidelity(pa.flatten(), pb.flatten())
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
