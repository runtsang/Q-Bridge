"""Quantum graph neural network utilities with a variational circuit backbone.

The QML counterpart now:
* Adds a `VariationalQNN` class that builds a layer‑wise circuit of
  parametrised single‑qubit rotations followed by a two‑qubit entangler.
* Provides a `train` method that optimises the rotation angles using
  the `pennylane` gradient engine.
* Exposes a `graph_from_states` that mirrors the classical version but
  uses the Qobj fidelity and supports optional spectral clustering on the
  state density matrices.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.numpy as npq
from pennylane import numpy as npnp

Tensor = npq.array
Array = np.ndarray


def _tensored_id(num_qubits: int) -> qml.QubitUnitary:
    return qml.Identity(num_qubits)


def _tensored_zero(num_qubits: int) -> qml.QubitUnitary:
    return qml.Identity(num_qubits)  # Zero projector not needed for qml


def _random_qubit_unitary(num_qubits: int) :  # pragma: no cover
    """Return a random unitary matrix using Pennylane's random generator."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    return qml.QubitUnitary(matrix, wires=range(num_qubits))


def _random_qubit_state(num_qubits: int) -> qml.StateVector:
    """Generate a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    return qml.StateVector(vec, wires=range(num_qubits))


def random_training_data(
    unitary: qml.QubitUnitary, samples: int
) -> List[Tuple[qml.StateVector, qml.StateVector]]:
    dataset = []
    num_qubits = len(unitary.wires)
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Create a random variational circuit with layer‑wise rotation + entangler."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qml.QubitUnitary]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qml.QubitUnitary] = []
        for output in range(num_outputs):
            # Single‑qubit rotation with a trainable angle
            rot = qml.RX(npq.random.uniform(-np.pi, np.pi), wires=output)
            # Entangler across current input qubits
            ent = qml.CNOT(wires=[output, (output + 1) % num_inputs])
            layer_ops.append(rot)
            layer_ops.append(ent)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitUnitary]],
    layer: int,
    input_state: qml.StateVector,
) -> qml.StateVector:
    """Apply a layer‑wise circuit to an input state and trace out unused qubits."""
    # Build a circuit for this layer
    dev = qml.device("default.qubit", wires=qnn_arch[layer])
    @qml.qnode(dev)
    def circuit(state):
        qml.StateVector(state, wires=range(qnn_arch[layer]))
        for gate in unitaries[layer]:
            qml.apply(gate)
        return qml.state()

    return circuit(input_state)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitUnitary]],
    samples: Iterable[Tuple[qml.StateVector, qml.StateVector]],
) -> List[List[qml.StateVector]]:
    """Run the variational circuit for each sample and collect intermediate states."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qml.StateVector, b: qml.StateVector) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qml.StateVector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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


class VariationalQNN:
    """Variational quantum neural network with trainable rotation angles."""

    def __init__(self, arch: Sequence[int]):
        self.arch = arch
        self.wires = arch[-1]
        self.params = npq.random.uniform(-np.pi, np.pi, size=(len(arch) - 1, self.wires))
        self.device = qml.device("default.qubit", wires=self.wires)

    def _circuit(self, state: qml.StateVector, params: Array) -> qml.StateVector:
        """Build a single‑layer circuit with the given rotation angles."""
        @qml.qnode(self.device)
        def circuit(state):
            qml.StateVector(state, wires=range(self.wires))
            for i, param in enumerate(params):
                qml.RX(param, wires=i)
            return qml.state()

        return circuit(state)

    def forward(self, state: qml.StateVector) -> qml.StateVector:
        """Apply all layers sequentially to the input state."""
        current = state
        for layer_idx in range(len(self.arch) - 1):
            current = self._circuit(current, self.params[layer_idx])
        return current

    def train(
        self,
        dataset: Iterable[Tuple[qml.StateVector, qml.StateVector]],
        lr: float = 0.01,
        epochs: int = 200,
    ) -> List[float]:
        """Train the variational circuit using gradient descent on the fidelity loss."""
        optimiser = qml.GradientDescentOptimizer(lr)
        losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for inp, tgt in dataset:
                def loss_fn(params):
                    pred = self.forward(inp)
                    return 1.0 - state_fidelity(pred, tgt)
                loss = optimiser.step(loss_fn, self.params)
                epoch_loss += loss
            losses.append(epoch_loss / len(dataset))
        return losses


def graph_from_states(
    states: Sequence[qml.StateVector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
    spectral_clusters: int | None = None,
) -> nx.Graph:
    """Create a weighted adjacency graph and optionally perform spectral clustering."""
    G = fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
    if spectral_clusters:
        # Compute Laplacian and eigenvectors for clustering
        L = nx.normalized_laplacian_matrix(G).todense()
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)[1 : spectral_clusters + 1]
        embed = eigvecs[:, idx]
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=spectral_clusters, n_init=10).fit_predict(embed)
        nx.set_node_attributes(G, {i: int(label) for i, label in enumerate(labels)}, "cluster")
    return G


__all__ = [
    "VariationalQNN",
    "feedforward",
    "graph_from_states",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
