import pennylane as qml
import numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

class GraphQNNGen052:
    """Quantum neural network using PennyLane.

    The network embeds classical features via ``AngleEmbedding`` and
    applies a variational circuit built from ``StronglyEntanglingLayers``.
    Public methods mirror the original GraphQNN seed to facilitate
    side‑by‑side experiments.
    """

    def __init__(self,
                 qnn_arch: Sequence[int],
                 dev_name: str = "default.qubit",
                 wires: int | None = None,
                 seed: int | None = None) -> None:
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[0]
        self.dev = qml.device(dev_name, wires=self.num_qubits)
        self.rng = np.random.default_rng(seed or 42)
        # Parameters for each variational layer
        self.params = [self.rng.standard_normal((self.num_qubits, 3))
                       for _ in range(len(self.arch) - 1)]

        @qml.qnode(self.dev, interface='autograd')
        def circuit(params, x):
            qml.AngleEmbedding(x, wires=range(self.num_qubits))
            for layer in params:
                qml.templates.StronglyEntanglingLayers(layer, wires=range(self.num_qubits))
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    @staticmethod
    def random_training_data(num_qubits: int,
                             samples: int,
                             seed: int | None = None) -> List[Tuple[np.ndarray, float]]:
        rng = np.random.default_rng(seed)
        dataset: List[Tuple[np.ndarray, float]] = []
        for _ in range(samples):
            x = rng.standard_normal(num_qubits)
            dataset.append((x, 0.0))  # target to be filled by ``random_network``
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int],
                       samples: int,
                       seed: int | None = None) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, float]], List[np.ndarray]]:
        rng = np.random.default_rng(seed)
        arch = list(qnn_arch)
        # Random target parameters for a variational circuit
        target_params = [rng.standard_normal((arch[0], 3))
                         for _ in range(len(arch) - 1)]

        # Helper to evaluate the target circuit
        dev_tgt = qml.device("default.qubit", wires=arch[0])

        @qml.qnode(dev_tgt, interface='numpy')
        def target_circuit(x):
            qml.AngleEmbedding(x, wires=range(arch[0]))
            for layer in target_params:
                qml.templates.StronglyEntanglingLayers(layer, wires=range(arch[0]))
            return qml.expval(qml.PauliZ(0))

        dataset = [(rng.standard_normal(arch[0]), target_circuit(x))
                   for x in rng.standard_normal((samples, arch[0]))]
        return arch, target_params, dataset, target_params

    def feedforward(self,
                    samples: Iterable[Tuple[np.ndarray, float]]) -> List[float]:
        """Return the network output for each input sample."""
        return [self.circuit(self.params, x) for x, _ in samples]

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(abs(a_norm.conj().dot(b_norm)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen052.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(self,
              samples: Iterable[Tuple[np.ndarray, float]],
              epochs: int = 200,
              lr: float = 0.01,
              verbose: bool = False) -> None:
        """Train the variational parameters to approximate the target outputs."""
        opt = qml.GradientDescentOptimizer(lr)
        for epoch in range(epochs):
            loss = 0.0
            for x, y in samples:
                pred = self.circuit(self.params, x)
                loss += (pred - y) ** 2
            loss /= len(samples)
            # Update parameters
            self.params = opt.step(self.params,
                                   lambda p: sum((self.circuit(p, x) - y) ** 2
                                                 for x, y in samples) / len(samples))
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.4f}")
