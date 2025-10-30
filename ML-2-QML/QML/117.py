"""GraphQNN: Quantum variational network for graph embeddings.

This module extends the original QML seed by implementing a
Pennylane variational circuit that acts on a classical graph
embedding vector.  The public functions mirror the seed API
(feedforward, random_network, random_training_data, state_fidelity,
fidelity_adjacency) while adding a ``GraphQNN`` class that
encapsulates the circuit and its parameters.
"""

import itertools
from typing import Iterable, List, Tuple, Sequence

import networkx as nx
import pennylane as qml
import numpy as np

Tensor = np.ndarray

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input_state, target_state) pairs."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        vec /= np.linalg.norm(vec)
        target = unitary @ vec
        dataset.append((vec, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit architecture and training data."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    layers: List[List[Tuple[str, int]]] = [[]]
    for _ in range(1, len(qnn_arch)):
        layer_ops: List[Tuple[str, int]] = []
        for qubit in range(num_qubits):
            layer_ops.append(("rx", qubit))
            layer_ops.append(("ry", qubit))
            layer_ops.append(("rz", qubit))
        layers.append(layer_ops)
    return list(qnn_arch), layers, training_data, target_unitary

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure states."""
    return np.abs(np.vdot(a, b)) ** 2

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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    """Pennylane variational circuit for graph embeddings.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture specification; the last element is the number of qubits.
    dev : qml.Device, optional
        PennyLane quantum device.  Defaults to ``default.qubit``.
    """

    def __init__(self, qnn_arch: Sequence[int], dev: qml.Device | None = None):
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[-1]
        self.dev = dev or qml.device("default.qubit", wires=self.num_qubits)
        self.params = np.random.randn(len(self.arch) - 1, self.num_qubits, 3)

        @qml.qnode(self.dev, interface="autograd", diff_method="adjoint")
        def circuit(state: np.ndarray, params: np.ndarray):
            qml.QubitStateVector(state, wires=range(self.num_qubits))
            for layer_idx, layer_params in enumerate(params):
                for qubit in range(self.num_qubits):
                    qml.RX(layer_params[qubit, 0], wires=qubit)
                    qml.RY(layer_params[qubit, 1], wires=qubit)
                    qml.RZ(layer_params[qubit, 2], wires=qubit)
                for qubit in range(self.num_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % self.num_qubits])
            return qml.state()

        self._circuit = circuit

    def feedforward(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """Run the circuit on each sample state."""
        stored: List[List[np.ndarray]] = []
        for input_state, _ in samples:
            layerwise = [input_state]
            current = input_state
            for layer_idx in range(len(self.arch) - 1):
                current = self._circuit(current, self.params[layer_idx : layer_idx + 1])
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    def encode_graph(self, embedding: np.ndarray) -> np.ndarray:
        """Return the circuit output for a given graph embedding."""
        return self._circuit(embedding, self.params)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
