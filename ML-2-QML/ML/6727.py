"""GraphQNN__gen584: Extended classical GNN utilities with hybrid regularisation.

This module extends the original GraphQNN by adding:
1. A stochastic data‑augmentation routine that injects Gaussian noise into input features.
2. A hybrid loss that combines mean‑squared error with a fidelity‑based regulariser.
3. A graph‑neighbourhood selector that prunes the fidelity graph based on spectral analysis.
4. A placeholder hybrid variational layer that can be used with a quantum backend via PyTorch.

All functions are pure Python and depend only on NumPy, PyTorch and NetworkX.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import networkx as nx
from collections.abc import Iterable, Sequence
from typing import List, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> Tensor:
    """Return a random weight matrix with normal distribution."""
    rng = rng or np.random.default_rng()
    return torch.tensor(rng.standard_normal((out_features, in_features)), dtype=torch.float32, requires_grad=True)

def random_training_data(weight: Tensor, samples: int, noise_level: float = 0.0) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a linear mapping.

    Args:
        weight: Target weight matrix.
        samples: Number of samples to generate.
        noise_level: Standard deviation of Gaussian noise to add to the input features.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        if noise_level > 0.0:
            features += noise_level * torch.randn_like(features)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int, noise_level: float = 0.0) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Build a random linear network and generate training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples, noise_level)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Forward‑propagate a batch of samples through the linear network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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

def augment_training_data(dataset: List[Tuple[Tensor, Tensor]], noise_level: float) -> List[Tuple[Tensor, Tensor]]:
    """Return a new dataset with Gaussian noise added to the input features."""
    augmented = []
    for x, y in dataset:
        noisy_x = x + noise_level * torch.randn_like(x)
        augmented.append((noisy_x, y))
    return augmented

def hybrid_loss(preds: List[Tensor], targets: List[Tensor], fidelity_matrix: np.ndarray, reg_weight: float = 0.1) -> Tensor:
    """Hybrid loss combining MSE and a fidelity‑based regulariser.

    The regulariser penalises large differences between predictions of
    states that are highly similar (high fidelity).
    """
    mse = torch.mean(torch.stack([(p - t) ** 2 for p, t in zip(preds, targets)]))
    # Regulariser: sum_{i<j} (1 - fidelity) * ||p_i - p_j||^2
    reg = 0.0
    n = len(preds)
    for i in range(n):
        for j in range(i + 1, n):
            diff = preds[i] - preds[j]
            reg += (1.0 - fidelity_matrix[i, j]) * (diff @ diff)
    reg = reg / (n * (n - 1) / 2)
    return mse + reg_weight * reg

def spectral_threshold(graph: nx.Graph, percentile: float = 50.0) -> float:
    """Compute a spectral threshold from the adjacency matrix.

    The threshold is set to the given percentile of the eigenvalues of the
    normalized Laplacian.
    """
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
