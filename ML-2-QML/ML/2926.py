"""Classical implementation of a hybrid graph‑based neural network.

The module extends the original GraphQNN utilities by adding
regression data generation, a small feed‑forward head and a
dataset wrapper that can be used with PyTorch DataLoaders.
The class `GraphQNNHybrid` can be instantiated with a
graph‑adjacency matrix (built by `fidelity_adjacency`) and
applies a classical neural network to node embeddings.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Random data generation helpers
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix initialized from a normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate inputs and targets for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a chain of random linear layers and a training set."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
#  Classical feed‑forward
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of activations for each sample."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

# --------------------------------------------------------------------------- #
#  Fidelity‑based graph utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two real vectors."""
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
#  Regression data and model
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a toy regression dataset from a superposition of angles."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns feature vectors and a target for regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """A small feed‑forward network that can be attached to the graph."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Hybrid graph‑to‑vector mapping
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """
    Classical wrapper that takes a graph produced by ``fidelity_adjacency`` and
    applies a neural network to node embeddings.  The embedding for a node is
    the concatenation of its activation vector across a random linear chain.
    """
    def __init__(self, qnn_arch: Sequence[int], graph: nx.Graph, device: torch.device | None = None):
        self.arch = list(qnn_arch)
        self.graph = graph
        self.device = device or torch.device("cpu")

        # Build a random linear chain used as a feature extractor
        _, self.weights, _, _ = random_network(self.arch, samples=1)
        for w in self.weights:
            w.requires_grad = False

        # Prepare a small neural head
        self.head = QModel(self.arch[-1]).to(self.device)

    def embed_graph(self, states: Sequence[Tensor]) -> Tensor:
        """Run the random linear chain on each node state and concatenate."""
        activations = feedforward(self.arch, self.weights, [(s, torch.tensor(0.0)) for s in states])
        # activations[0] is the input, activations[-1] the final output
        node_vectors = [a[-1] for a in activations]
        return torch.stack(node_vectors, dim=0, device=self.device)

    def forward(self, states: Sequence[Tensor]) -> Tensor:
        """Return a scalar prediction for the whole graph."""
        node_emb = self.embed_graph(states)
        # Aggregate by averaging over nodes, then feed through head
        graph_repr = node_emb.mean(dim=0, keepdim=True)
        return self.head(graph_repr)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "GraphQNNHybrid",
]
