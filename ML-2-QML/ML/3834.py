"""Hybrid classical graph neural network with regression head.

This module merges concepts from the original GraphQNN (classical
feed‑forward + fidelity graph) and QuantumRegression (regression
dataset and model).  It exposes a single ``GraphQNNGen132`` class that
performs message passing over a fidelity‑derived adjacency matrix
followed by a small MLP head.
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


# --------------------------------------------------------------------------- #
#  Utility functions – data generation and fidelity graph construction
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: torch.Tensor,
    samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic regression targets from a linear weight matrix."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Build a random linear network and a corresponding target weight."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Return activations for each sample through the linear network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two classical feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph where edges are added by fidelity thresholds."""
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
#  Dataset utilities – generate random graph samples and regression targets
# --------------------------------------------------------------------------- #
class GraphRegressionDataset(Dataset):
    """Random graph + regression target dataset.

    Each sample consists of node features and a target obtained by applying a
    random linear transformation to the aggregated node features.
    """
    def __init__(self, samples: int, num_nodes: int, in_features: int):
        self.graphs, self.targets = self._create_samples(samples, num_nodes, in_features)

    @staticmethod
    def _create_samples(samples: int, num_nodes: int, in_features: int):
        graphs: List[nx.Graph] = []
        targets: List[torch.Tensor] = []
        # Random linear map used for all samples (shared for consistency)
        weight = torch.randn(in_features, 1)
        for _ in range(samples):
            G = nx.gnp_random_graph(num_nodes, 0.3)
            # Assign random node features
            for n in G.nodes():
                G.nodes[n]["feat"] = torch.randn(in_features)
            # Aggregate features (mean) and compute target
            feats = torch.stack([G.nodes[n]["feat"] for n in G.nodes()])
            agg = feats.mean(dim=0)
            target = weight.t() @ agg
            graphs.append(G)
            targets.append(target.squeeze())
        return graphs, targets

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.graphs)

    def __getitem__(self, index: int):  # type: ignore[override]
        G = self.graphs[index]
        node_feats = torch.stack([G.nodes[n]["feat"] for n in G.nodes()])
        return {"graph": G, "node_feats": node_feats, "target": self.targets[index]}


# --------------------------------------------------------------------------- #
#  Main model – classical graph neural network
# --------------------------------------------------------------------------- #
class GraphQNNGen132(nn.Module):
    """Hybrid GNN that uses fidelity‑based adjacency for message passing."""
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_features)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Classical feed‑forward
        h = torch.relu(self.linear1(node_feats))
        h = torch.relu(self.linear2(h))
        # Message passing via fidelity graph
        h = torch.matmul(adj, h)
        return self.out(h)


__all__ = [
    "GraphQNNGen132",
    "GraphRegressionDataset",
    "fidelity_adjacency",
    "random_network",
    "feedforward",
]
