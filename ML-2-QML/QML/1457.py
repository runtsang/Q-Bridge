"""Graph‑based quantum neural network utilities enriched with variational training.

The module keeps the same public interface as the classical version but
provides a parameterised PennyLane circuit for each layer.  A
variational optimisation routine is included, and the fidelity‑based
adjacency graph is extended with community detection similar to the
classical side.  All imports are explicit and the code is fully
stand‑alone.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray

# --------------------------------------------------------------------------- #
# 1. Variational primitives
# --------------------------------------------------------------------------- #
def _parametric_layer(params: np.ndarray) -> None:
    """Apply a simple Ry‑rotation + CNOT ladder on the qubits.

    Parameters
    ----------
    params : np.ndarray
        Rotation angles for each qubit.  Length of ``params`` equals the
        number of qubits in the layer.
    """
    for q, angle in enumerate(params):
        qml.Ry(angle, wires=q)
    for q in range(len(params) - 1):
        qml.CNOT(wires=[q, q + 1])


# --------------------------------------------------------------------------- #
# 2. Random network construction
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[Tensor, Tensor]], np.ndarray]:
    """Generate a random parameterised QNN and training data.

    Parameters
    ----------
    qnn_arch : List[int]
        Layer widths including input and output qubits.
    samples : int
        Number of training samples to generate for the target unitary.

    Returns
    -------
    Tuple[List[int], List[List[np.ndarray]], List[Tuple[Tensor, Tensor]], np.ndarray]
        Architecture, list of per‑layer parameter arrays, training data, and
        target unitary parameters.
    """
    # Target unitary parameters for the output layer
    target_params = np.random.randn(qnn_arch[-1])

    # Training data: random input states and target states
    training_data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = np.random.randn(2 ** qnn_arch[0]) + 1j * np.random.randn(2 ** qnn_arch[0])
        state /= np.linalg.norm(state)
        target_state = _apply_unitary(target_params, state, qnn_arch[-1])
        training_data.append((state, target_state))

    # Per‑layer parameters
    layer_params: List[List[np.ndarray]] = []
    for num_qubits in qnn_arch[1:-1]:
        params = np.random.randn(num_qubits)
        layer_params.append([params])

    return qnn_arch, layer_params, training_data, target_params


def _apply_unitary(params: np.ndarray, state: Tensor, num_qubits: int) -> Tensor:
    """Apply a single‑qubit Ry‑rotation + CNOT ladder unitary defined by ``params``."""
    def circuit(state_in, all_params):
        qml.StatePrep(state_in, wires=range(len(state_in)))
        _parametric_layer(all_params[0])
        return qml.State()

    dev = qml.device("default.qubit", wires=2 ** num_qubits)
    qnode = qml.QNode(circuit, dev)
    return qnode(state, [params])


# --------------------------------------------------------------------------- #
# 3. Random training data
# --------------------------------------------------------------------------- #
def random_training_data(unitary_params: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training data for a given unitary defined by ``unitary_params``."""
    training_data: List[Tuple[Tensor, Tensor]] = []
    num_qubits = len(unitary_params)
    for _ in range(samples):
        state = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
        state /= np.linalg.norm(state)
        target = _apply_unitary(unitary_params, state, num_qubits)
        training_data.append((state, target))
    return training_data


# --------------------------------------------------------------------------- #
# 4. Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    layer_params: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the QNN on a collection of input states.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the network.
    layer_params : Sequence[Sequence[np.ndarray]]
        Parameters for each layer; each inner list contains
        the parameters for that layer.
    samples : Iterable[Tuple[Tensor, Tensor]]
        ``(input_state, target_state)`` pairs; the target is ignored
        during inference.

    Returns
    -------
    List[List[Tensor]]
        For each sample, a list containing the input and output state.
    """
    dev = qml.device("default.qubit", wires=2 ** qnn_arch[0])

    def circuit(state_in, all_params):
        qml.StatePrep(state_in, wires=range(len(state_in)))
        for params in all_params:
            _parametric_layer(params)
        return qml.State()

    qnode = qml.QNode(circuit, dev)

    all_params = [p[0] for p in layer_params]
    stored_states: List[List[Tensor]] = []
    for state, _ in samples:
        output_state = qnode(state, all_params)
        stored_states.append([state, output_state])
    return stored_states


# --------------------------------------------------------------------------- #
# 5. Fidelity helpers
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.  Communities are
    assigned using the greedy modularity algorithm.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)

    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    for idx, comm in enumerate(communities):
        for node in comm:
            graph.nodes[node]["community"] = idx
    return graph


# --------------------------------------------------------------------------- #
# 6. Variational training routine
# --------------------------------------------------------------------------- #
def train_variational_qnn(
    qnn_arch: Sequence[int],
    layer_params: List[List[np.ndarray]],
    training_data: List[Tuple[Tensor, Tensor]],
    lr: float,
    epochs: int,
    optimizer_cls=qml.GradientDescentOptimizer,
) -> List[List[np.ndarray]]:
    """Optimize the parameters of the QNN to match the target states.

    Uses PennyLane's built‑in gradient descent optimizer.  The loss is
    the mean squared error between the output state and the target
    state, expressed as the fidelity loss ``1 - |<ψ|φ>|²``.
    """
    opt = optimizer_cls(lr)
    params = [p[0] for p in layer_params]

    dev = qml.device("default.qubit", wires=2 ** qnn_arch[0])

    def circuit(state_in, all_params):
        qml.StatePrep(state_in, wires=range(len(state_in)))
        for params in all_params:
            _parametric_layer(params)
        return qml.State()

    qnode = qml.QNode(circuit, dev)

    for _ in range(epochs):
        for state, target in training_data:
            def loss_fn(all_params):
                out = qnode(state, all_params)
                fid = abs(np.vdot(out, target)) ** 2
                return 1 - fid
            params = opt.step(loss_fn, params)

    return [[p] for p in params]


# --------------------------------------------------------------------------- #
# 7. Spectral clustering harness
# --------------------------------------------------------------------------- #
def spectral_clustering(graph: nx.Graph, n_clusters: int) -> List[int]:
    """Return a list assigning each node to one of ``n_clusters`` clusters
    based on the graph Laplacian.  The function uses NetworkX's
    ``spectral_clustering`` routine via SciPy.
    """
    from sklearn.cluster import SpectralClustering

    A = nx.to_numpy_array(graph, weight="weight")
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )
    labels = clustering.fit_predict(A)
    return labels.tolist()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_variational_qnn",
    "spectral_clustering",
]
