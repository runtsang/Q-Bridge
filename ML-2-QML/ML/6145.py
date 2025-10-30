"""GraphQNN_Gen – a hybrid graph neural network with trainable edge weights.

This module extends the original GraphQNN by adding an
edge‑aware message‑passing layer that learns a weight matrix
for each adjacency edge.  The architecture is still
expressible as a sequence of layer sizes, but the forward
pass now collects the adjacency‑based messages and
applies a learnable transformation before the activation.
This allows the network to capture structural
information that the original feed‑forward only model
could not.  The module is fully classical and uses
PyTorch for the trainable parameters.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for the target linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

# --------------------------------------------------------------------------- #
# Graph‑aware random network creation
# --------------------------------------------------------------------------- #
def random_network(
    qnn_arch: Sequence[int],
    graph: Optional[nx.Graph] = None,
    samples: int = 100,
) -> Tuple[List[int], List[Tensor], List[Optional[Tensor]], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random weight matrix for each layer. If a graph is supplied,
    a trainable edge‑weight vector is created for every layer.
    """
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    edge_weights: List[Optional[Tensor]] = []
    if graph is not None:
        num_edges = graph.number_of_edges()
        for _ in weights[:-1]:
            edge_weights.append(torch.randn(num_edges, 1, dtype=torch.float32))
        edge_weights.append(None)  # last layer has no outgoing edges
    else:
        edge_weights = [None] * len(weights)

    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, edge_weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# Forward propagation with message passing
# --------------------------------------------------------------------------- #
def _adjacency_message(
    graph: nx.Graph,
    features: Tensor,
) -> Tensor:
    """Aggregate messages from neighbours using the graph adjacency matrix."""
    adj = nx.to_numpy_array(graph, dtype=float)
    return torch.tensor(adj @ features.numpy(), dtype=features.dtype)

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    edge_weights: Sequence[Optional[Tensor]],
    samples: Iterable[Tuple[Tensor, Tensor]],
    graph: Optional[nx.Graph] = None,
) -> List[List[Tensor]]:
    """Run a graph‑aware feed‑forward pass."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for w, ew in zip(weights, edge_weights):
            if ew is not None and graph is not None:
                # weighted message passing
                msg = _adjacency_message(graph, current)
                # apply edge weights per neighbour
                # reshape edge weights to broadcast over nodes
                num_nodes = graph.number_of_nodes()
                adj = nx.to_numpy_array(graph, dtype=float)
                w_adj = torch.tensor(adj * ew.squeeze().numpy(), dtype=current.dtype)
                msg = w_adj @ current
                current = msg
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
