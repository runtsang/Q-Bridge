import pennylane as qml
import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

class GraphQNN:
    """
    Quantum graph‑based neural network using Pennylane.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes. The last element defines the number of qubits.
    residual : bool, default=True
        If ``True`` a residual connection adds the input state to the
        output of each layer.
    """

    def __init__(self, arch: Sequence[int], residual: bool = True):
        self.arch = list(arch)
        self.residual = residual
        self.num_qubits = arch[-1]
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        # One unitary per layer (acting on all qubits)
        self.unitaries: List[np.ndarray] = [self._random_unitary(self.num_qubits)
                                            for _ in arch[1:]]

    # ------------------------------------------------------------------
    #  Random unitary generator
    # ------------------------------------------------------------------
    @staticmethod
    def _random_unitary(num_qubits: int) -> np.ndarray:
        dim = 2 ** num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return q

    # ------------------------------------------------------------------
    #  Training data generator
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(target_unitary: np.ndarray, samples: int
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate random input states and the corresponding target states
        obtained by applying ``target_unitary``.
        """
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        dim = target_unitary.shape[0]
        for _ in range(samples):
            vec = np.random.randn(dim) + 1j * np.random.randn(dim)
            vec /= np.linalg.norm(vec)
            target = target_unitary @ vec
            dataset.append((vec, target))
        return dataset

    # ------------------------------------------------------------------
    #  Random network constructor
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """
        Create a random network together with a training dataset.
        """
        # Random unitary for each layer
        unitaries = [GraphQNN._random_unitary(arch[-1]) for _ in arch[1:]]
        target_unitary = unitaries[-1]
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        return arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------
    #  Forward propagation
    # ------------------------------------------------------------------
    def _apply_layer(self, state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """
        Apply a single unitary to a state vector.
        """
        return unitary @ state

    def feedforward(self,
                    samples: Iterable[Tuple[np.ndarray, np.ndarray]]
                   ) -> List[List[np.ndarray]]:
        """
        Run a batch of input states through the network.

        Returns a list of state lists per sample; each list contains the
        input followed by the output of every layer.
        """
        stored: List[List[np.ndarray]] = []
        for state, _ in samples:
            activations: List[np.ndarray] = [state]
            current = state
            for unitary in self.unitaries:
                current = self._apply_layer(current, unitary)
                if self.residual:
                    current = current + activations[-1]
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------
    #  Fidelity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Return the squared overlap between two pure state vectors.
        """
        return abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph where edges represent fidelity between states.
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

    # ------------------------------------------------------------------
    #  Loss based on graph adjacency
    # ------------------------------------------------------------------
    def fidelity_loss(self,
                      outputs: List[np.ndarray],
                      graph: nx.Graph) -> float:
        """
        Compute a graph‑weighted loss: sum (1 - fidelity) over all edges.
        """
        loss = 0.0
        for i, j, data in graph.edges(data=True):
            fi = outputs[i]
            fj = outputs[j]
            fid = self.state_fidelity(fi, fj)
            loss += (1.0 - fid) * data.get('weight', 1.0)
        return loss / graph.number_of_edges()

    __all__ = [
        "GraphQNN",
    ]
