"""Graph‑based quantum neural network using Pennylane.

The implementation mirrors the classical counterpart but replaces the
GCN/MLP layers with a variational circuit that encodes graph structure
into a quantum state.  Fidelity‑based adjacency graphs are computed
using quantum state overlap.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np

Tensor = qml.QubitStateVector
GraphData = Tuple[np.ndarray, np.ndarray]  # (node_features, adjacency_matrix)


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary via QR decomposition."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Random pure state on `num_qubits`."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


class GraphQNNGen462:
    """Variational QNN that processes graph data encoded into qubits."""

    def __init__(
        self,
        arch: Sequence[int],
        num_qubits: int,
        dev: qml.Device | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        arch: Sequence[int]
            Number of variational layers (each uses random rotations).
        num_qubits: int
            Number of qubits used to encode the graph.
        dev: pennylane.Device, optional
            Pennylane device; defaults to a default qubit simulator.
        seed: int, optional
            Random seed for reproducibility.
        """
        self.arch = list(arch)
        self.num_qubits = num_qubits
        self.seed = seed
        if dev is None:
            dev = qml.device("default.qubit", wires=num_qubits)
        self.dev = dev

        # Pre‑generate a random target unitary for training
        self.target_unitary = _random_qubit_unitary(num_qubits)

    # ------------------------------------------------------------------
    # Circuit definition
    # ------------------------------------------------------------------

    def _encode_graph(self, node_features: np.ndarray, adjacency: np.ndarray):
        """Encode graph into qubit states via a simple feature map."""
        for i, feat in enumerate(node_features):
            qml.RY(feat, wires=i)
        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[1]):
                if adjacency[i, j]:
                    qml.CRY(np.pi / 4, wires=[i, j])

    def circuit(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Run the variational circuit and return the final statevector."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit_fn():
            self._encode_graph(node_features, adjacency)
            for _ in range(len(self.arch)):
                for i in range(self.num_qubits):
                    angle = np.random.uniform(0, 2 * np.pi)
                    qml.RY(angle, wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        return circuit_fn()

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate training pairs (input state, target state)."""
        dataset = []
        num_qubits = int(np.log2(unitary.shape[0]))
        for _ in range(samples):
            state = _random_qubit_state(num_qubits)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        """Create a random target unitary and training data."""
        target_unitary = _random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNNGen462.random_training_data(target_unitary, samples)
        return qnn_arch, [target_unitary], training_data, target_unitary

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        unitaries: Sequence[np.ndarray],
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """Run the circuit on a collection of input states."""
        outputs: List[List[np.ndarray]] = []
        for input_state, _ in samples:
            adjacency = np.eye(len(input_state))
            layerwise = [input_state]
            state = input_state
            for _ in range(len(qnn_arch)):
                state = self.circuit(state, adjacency)
                layerwise.append(state)
            outputs.append(layerwise)
        return outputs

    # ------------------------------------------------------------------
    # Fidelity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared absolute overlap between two pure states."""
        return float(np.abs(a @ b.conj()) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct weighted graph from quantum state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen462.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
