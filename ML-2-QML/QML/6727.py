"""GraphQNN__gen584: Quantum version of the extended GraphQNN module.

This module implements the same utilities as the classical counterpart but
uses PennyLane to build parameterised quantum circuits.  It also provides
data‑augmentation, a hybrid loss and spectral graph pruning.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import networkx as nx
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import pennylane as qml

Tensor = torch.Tensor

def _random_qubit_unitary(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    rng = rng or np.random.default_rng()
    dim = 2 ** num_qubits
    matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(matrix)
    return q

def _random_qubit_state(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random pure state vector of size 2**num_qubits."""
    rng = rng or np.random.default_rng()
    dim = 2 ** num_qubits
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(unitary: np.ndarray, samples: int, noise_level: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training data for a unitary mapping.

    Args:
        unitary: Target unitary matrix.
        samples: Number of samples to generate.
        noise_level: Standard deviation of Gaussian noise to add to the input state.
    """
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = _random_qubit_state(int(np.log2(dim)))
        if noise_level > 0.0:
            noise = _random_qubit_state(int(np.log2(dim))) * noise_level
            state = state + noise
            state /= np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int, noise_level: float = 0.0) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build a random multi‑layer unitary network and generate training data."""
    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer - 1]
        out_q = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples, noise_level)
    return list(qnn_arch), unitaries, training_data, target_unitary

def _apply_unitary(state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Apply a unitary to a state vector."""
    return unitary @ state

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Forward‑propagate a batch of samples through the unitary network."""
    stored: List[List[np.ndarray]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            # multiply all gates in this layer sequentially
            layer_unitary = unitaries[layer][0]
            for gate in unitaries[layer][1:]:
                layer_unitary = gate @ layer_unitary
            current = _apply_unitary(current, layer_unitary)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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

# --------------------------------------------------------------------------- #
# New functionality
# --------------------------------------------------------------------------- #

def augment_training_data(dataset: List[Tuple[np.ndarray, np.ndarray]], noise_level: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a new dataset with Gaussian noise added to the input state."""
    augmented = []
    for state, target in dataset:
        noise = _random_qubit_state(int(np.log2(state.shape[0]))) * noise_level
        noisy_state = state + noise
        noisy_state /= np.linalg.norm(noisy_state)
        augmented.append((noisy_state, target))
    return augmented

def hybrid_loss(preds: List[np.ndarray], targets: List[np.ndarray], fidelity_matrix: np.ndarray, reg_weight: float = 0.1) -> Tensor:
    """Hybrid loss combining MSE of state vectors and a fidelity‑based regulariser."""
    # Convert to torch tensors
    preds_t = torch.tensor(np.stack(preds), dtype=torch.complex64)
    targets_t = torch.tensor(np.stack(targets), dtype=torch.complex64)
    mse = torch.mean((preds_t - targets_t) ** 2).real
    # Regulariser: sum_{i<j} (1 - fidelity) * ||p_i - p_j||^2
    reg = 0.0
    n = len(preds)
    for i in range(n):
        for j in range(i + 1, n):
            diff = preds_t[i] - preds_t[j]
            reg += (1.0 - fidelity_matrix[i, j]) * (diff @ diff).real
    reg = reg / (n * (n - 1) / 2)
    return mse + reg_weight * reg

def spectral_threshold(graph: nx.Graph, percentile: float = 50.0) -> float:
    """Compute a spectral threshold from the adjacency matrix."""
    adj = nx.to_numpy_array(graph)
    lap = np.diag(adj.sum(axis=1)) - adj
    eigvals = np.linalg.eigvalsh(lap)
    return float(np.percentile(eigvals, percentile))

def prune_fidelity_graph(graph: nx.Graph, threshold: float | None = None, percentile: float = 50.0) -> nx.Graph:
    """Return a subgraph containing edges above a spectral‑based threshold."""
    if threshold is None:
        threshold = spectral_threshold(graph, percentile)
    sub = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if data.get("weight", 0.0) >= threshold:
            sub.add_edge(u, v, weight=data["weight"])
    return sub

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "augment_training_data",
    "hybrid_loss",
    "spectral_threshold",
    "prune_fidelity_graph",
]
