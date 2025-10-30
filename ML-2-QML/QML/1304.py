import pennylane as qml
import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

def random_training_data(target_unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training pairs (state, target_state) for a fixed unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        target = target_unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a toy variational circuit and training data."""
    num_qubits = qnn_arch[-1]
    dim = 2 ** num_qubits
    target_unitary = qml.math.random_unitary(dim)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), target_unitary, training_data


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap of two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNGen:
    """
    Variational graph‑QNN with a Laplacian regulariser.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Widths of the layers (number of qubits per layer).
    lr : float, optional
        Optimiser step size.
    epochs : int, optional
        Number of optimisation epochs.
    reg_weight : float, optional
        Weight of the Laplacian regulariser.
    device : str, optional
        Pennylane device name.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        lr: float = 0.01,
        epochs: int = 200,
        reg_weight: float = 0.01,
        device: str = "default.qubit",
    ):
        self.arch = list(qnn_arch)
        self.lr = lr
        self.epochs = epochs
        self.reg_weight = reg_weight
        self.device = qml.device(device, wires=self.arch[-1])

        # One rotation per qubit per layer
        self.params = np.random.randn(len(self.arch) - 1, self.arch[-1], 3)
        self.optimizer = qml.AdamOptimizer(stepsize=self.lr)

    def _circuit(self, params, wires):
        for layer, layer_params in enumerate(params):
            for q, (rx, ry, rz) in enumerate(layer_params):
                qml.Rot(rx, ry, rz, wires=[wires[q]])
            if layer < len(params) - 1:
                for q in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[q], wires[q + 1]])
        return qml.state()

    def cost(self, params, target_state):
        state = self._circuit(params, self.device.wires)
        return 1 - state_fidelity(state, target_state)

    def train(self, dataset: Iterable[Tuple[np.ndarray, np.ndarray]]):
        last_loss = None
        for _ in range(self.epochs):
            for state, target in dataset:
                self.params, loss = self.optimizer.step_and_cost(
                    lambda p: self.cost(p, target), self.params
                )
                last_loss = loss
        return last_loss

    def run(self, dataset: Iterable[Tuple[np.ndarray, np.ndarray]]) -> dict:
        final_loss = self.train(dataset)

        # Fidelity against a random target state
        random_state = np.random.randn(2 ** self.arch[-1]) + 1j * np.random.randn(2 ** self.arch[-1])
        random_state /= np.linalg.norm(random_state)
        final_state = self._circuit(self.params, self.device.wires)
        fid = state_fidelity(final_state, random_state)

        # Graph diagnostics from outputs across samples
        outputs = [self._circuit(self.params, self.device.wires) for _ in dataset]
        graph = fidelity_adjacency(outputs, threshold=0.8)
        lap = nx.laplacian_matrix(graph).astype(float)
        eigs = np.linalg.eigvalsh(lap.todense())

        return {
            "loss": float(final_loss),
            "fidelity": float(fid),
            "laplacian_spectrum": eigs.tolist(),
        }


__all__ = [
    "GraphQNNGen",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
