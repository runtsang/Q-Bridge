"""Graph-based neural network utilities with batched support and adjacency analysis.

The module intentionally mirrors the QML API but adds mini‑batch
processing, L2‑regularized weight updates, and a simple spectral
clustering harness.  The implementation is fully classical and relies
on PyTorch for tensor arithmetic and NetworkX for graph handling.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Weight helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)


def _l2_norm(weight: Tensor) -> float:
    """Return the L2 norm of a weight tensor."""
    return float(torch.norm(weight))


# --------------------------------------------------------------------------- #
# 2. Data generation
# --------------------------------------------------------------------------- #
def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset for the target weight matrix.

    Parameters
    ----------
    weight : torch.Tensor
        Target linear map of shape (output, input).
    samples : int
        Number of examples to generate.

    Returns
    -------
    List[Tuple[Tensor, Tensor]]
        ``(features, target)`` pairs where ``features`` has shape
        (input,) and ``target`` = ``weight @ features``.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random feed‑forward network together with training data.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer widths including input and output nodes.
    samples : int
        Number of training samples to generate for the last layer.

    Returns
    -------
    Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]
        Architecture, list of weight tensors, training data, and target weight.
    """
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
# 3. Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute activations for a batch of samples.

    The function returns a list of lists where each sublist contains
    the activations of all layers for one sample.  Activations are
    computed using ``torch.tanh``.
    """
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


# --------------------------------------------------------------------------- #
# 4. Fidelity helpers
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared cosine similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive
    weight 1.  When ``secondary`` is provided, fidelities between
    ``secondary`` and ``threshold`` are added with ``secondary_weight``.
    Additionally, the graph is enriched with community labels obtained
    via the greedy modularity algorithm.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)

    # Community detection
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    for idx, comm in enumerate(communities):
        for node in comm:
            graph.nodes[node]["community"] = idx
    return graph


# --------------------------------------------------------------------------- #
# 5. Spectral clustering harness
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


# --------------------------------------------------------------------------- #
# 6. Simple training routine
# --------------------------------------------------------------------------- #
def train_linear_model(
    qnn_arch: Sequence[int],
    weights: List[Tensor],
    training_data: List[Tuple[Tensor, Tensor]],
    lr: float,
    epochs: int,
    l2_lambda: float = 0.0,
) -> List[Tensor]:
    """Fine‑tune the network on ``training_data`` using gradient descent.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the network.
    weights : List[Tensor]
        Current weight tensors to be updated in‑place.
    training_data : List[Tuple[Tensor, Tensor]]
        ``(input, target)`` pairs.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    l2_lambda : float
        Optional L2 regularisation coefficient.

    Returns
    -------
    List[Tensor]
        Updated weight tensors.
    """
    for _ in range(epochs):
        for features, target in training_data:
            # Forward
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)

            # Loss
            loss = torch.nn.functional.mse_loss(activations[-1], target)
            if l2_lambda > 0:
                loss += l2_lambda * sum(_l2_norm(w) for w in weights)

            # Backward
            loss.backward()
            for w in weights:
                w.data -= lr * w.grad.data
                w.grad.zero_()
    return weights


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "spectral_clustering",
    "train_linear_model",
]
