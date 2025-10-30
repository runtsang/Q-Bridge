import numpy as np
import networkx as nx
import itertools
import pennylane as qml
import pennylane.numpy as pnp
from typing import Iterable, List

class HybridKernelGraphQNN:
    """Quantum‑graph hybrid module using Pennylane.

    Provides a variational quantum kernel, a classical RBF kernel for comparison,
    and graph‑neural‑network style state propagation.  The class builds
    weighted graphs from quantum state fidelities and supports end‑to‑end
    differentiable training with Pennylane's autograd.
    """

    # -------------------- Classical RBF kernel --------------------
    class _ClassicalRBFKernel:
        def __init__(self, gamma: float = 1.0):
            self.gamma = gamma

        def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            diff = x[:, None, :] - y[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            return np.exp(-self.gamma * d2)

    # -------------------- Quantum kernel --------------------
    class _QuantumKernel:
        def __init__(self, n_wires: int):
            self.n_wires = n_wires
            self.dev = qml.device("default.qubit", wires=n_wires)

        def _circuit(self, x: np.ndarray):
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()

        @qml.qnode
        def _state(self, x: np.ndarray):
            return self._circuit(x)

        def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
            psi_x = self._state(x)
            psi_y = self._state(y)
            return np.abs(np.vdot(psi_x, psi_y)) ** 2

    # -------------------- Graph utilities --------------------
    @staticmethod
    def _random_unitary(n_qubits: int) -> np.ndarray:
        """Return a random unitary matrix of dimension 2**n_qubits."""
        dim = 2 ** n_qubits
        random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(random_matrix)
        return q

    @staticmethod
    def _random_training_data(unitary: np.ndarray, samples: int) -> List[tuple[np.ndarray, np.ndarray]]:
        dataset: List[tuple[np.ndarray, np.ndarray]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            state = np.random.randn(dim) + 1j * np.random.randn(dim)
            state = state / np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def _random_network(qnn_arch: List[int], samples: int):
        n_wires = max(qnn_arch)
        unitaries: List[np.ndarray] = []
        for _ in range(len(qnn_arch) - 1):
            unitaries.append(HybridKernelGraphQNN._random_unitary(n_wires))
        target_unitary = HybridKernelGraphQNN._random_unitary(n_wires)
        training_data = HybridKernelGraphQNN._random_training_data(target_unitary, samples)
        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _feedforward(
        qnn_arch: List[int],
        unitaries: List[np.ndarray],
        samples: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        stored: List[List[np.ndarray]] = []
        for state, _ in samples:
            layerwise = [state]
            current = state
            for U in unitaries:
                current = U @ current
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        return np.abs(np.vdot(a, b)) ** 2

    @staticmethod
    def _fidelity_adjacency(
        states: List[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridKernelGraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -------------------- Public API --------------------
    def __init__(self, gamma: float = 1.0, n_wires: int | None = None):
        self.gamma = gamma
        self.classical_kernel = self._ClassicalRBFKernel(gamma)
        self.n_wires = n_wires or 4
        self.quantum_kernel = self._QuantumKernel(self.n_wires)

    def classical_kernel_matrix(self, a: List[np.ndarray], b: List[np.ndarray]) -> np.ndarray:
        return np.array([[self.classical_kernel(x, y).item() for y in b] for x in a])

    def quantum_kernel_matrix(self, a: List[np.ndarray], b: List[np.ndarray]) -> np.ndarray:
        return np.array([[self.quantum_kernel.kernel(x, y) for y in b] for x in a])

    def feedforward(self, qnn_arch, unitaries, samples):
        return self._feedforward(qnn_arch, unitaries, samples)

    def fidelity_adjacency(self, states, threshold, secondary=None, secondary_weight=0.5):
        return self._fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def random_network(self, qnn_arch, samples):
        return self._random_network(qnn_arch, samples)

    def random_training_data(self, unitary, samples):
        return self._random_training_data(unitary, samples)

__all__ = [
    "HybridKernelGraphQNN",
    "_ClassicalRBFKernel",
    "_QuantumKernel",
    "_random_unitary",
    "random_network",
    "random_training_data",
    "_feedforward",
    "state_fidelity",
    "_fidelity_adjacency",
]
