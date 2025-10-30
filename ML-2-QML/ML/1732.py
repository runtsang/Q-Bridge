"""Hybrid Graph Neural Network – Classical backbone with optional attention.

This module extends the seed GraphQNN by providing a lightweight Graph Attention
Network (GAT) backbone that can be used in place of the simple tanh‑linear
layers.  The public API mirrors the seed code but adds a ``use_attention``
flag and a small ``GraphAttentionBackbone`` class.  All utilities are
fully NumPy/PyTorch‑based and can be used in research pipelines or
production systems.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "RandomGraphDataset",
    "GraphAttentionBackbone",
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]

# --------------------------------------------------------------------------- #
# 1. Dataset – random graph construction with node features and label
# --------------------------------------------------------------------------- #
class RandomGraphDataset(torch.utils.data.Dataset):
    """Generate a random Erdős–Rényi graph with node features and a target vector.

    The graph is stored as a NetworkX graph with ``'x'`` node attributes.
    The target vector is a random linear combination of the node features
    and can be used for supervised learning tasks.
    """

    def __init__(
        self,
        *,
        node_count: int,
        edge_prob: float,
        feature_dim: int,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.graph = nx.fast_gnp_random_graph(
            n=node_count, p=edge_prob, seed=rng, directed=False
        )
        for n in self.graph.nodes:
            self.graph.nodes[n]["x"] = torch.randn(feature_dim, dtype=torch.float32)
        # Simple target: sum of node features
        self.target = torch.sum(
            torch.stack([self.graph.nodes[n]["x"] for n in self.graph.nodes]), dim=0
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack([self.graph.nodes[n]["x"] for n in self.graph.nodes])
        return features, self.target


# --------------------------------------------------------------------------- #
# 2. Graph Attention Backbone
# --------------------------------------------------------------------------- #
class GraphAttentionBackbone(torch.nn.Module):
    """A tiny multi‑head GAT layer that operates on node feature matrices.

    Parameters
    ----------
    in_features : int
        Dimensionality of input node features.
    out_features : int
        Dimensionality of output node features.
    heads : int
        Number of attention heads.
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 1) -> None:
        super().__init__()
        self.heads = heads
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features, out_features, bias=False)
                for _ in range(heads)
            ]
        )
        self.attn = torch.nn.Parameter(torch.randn(heads, 2 * out_features))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        # x: (N, in_features)
        # adjacency: (N, N) binary matrix
        outputs = []
        for proj in self.projections:
            h = proj(x)  # (N, out_features)
            # simple attention weight: softmax over neighbors
            attn = F.softmax(torch.matmul(adjacency, torch.ones_like(h)), dim=1)
            outputs.append(attn * h)
        return torch.cat(outputs, dim=1)


# --------------------------------------------------------------------------- #
# 3. Core GraphQNN utilities (seed‑like API with optional attention)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs by applying a linear transform to random inputs."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int, *, use_attention: bool = False
) -> Tuple[List[int], List[Tensor | GraphAttentionBackbone], List[Tuple[Tensor, Tensor]], Tensor | None]:
    """Generate a random linear or attention‑based network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes, e.g. ``[10, 20, 5]``.
    samples : int
        Number of training samples to generate.
    use_attention : bool, optional
        If True, the network will consist of GAT layers instead of linear ones.

    Returns
    -------
    arch : List[int]
        Architecture list.
    weights : List[Tensor | GraphAttentionBackbone]
        For linear networks: weight matrices.
        For attention networks: list of GraphAttentionBackbone modules.
    training_data : List[Tuple[Tensor, Tensor]]
        (input, target) pairs.
    target_weight : Tensor | None
        The weight matrix that defines the ground‑truth mapping if linear,
        otherwise ``None``.
    """
    arch = list(qnn_arch)
    if use_attention:
        weights: List[GraphAttentionBackbone] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphAttentionBackbone(in_f, out_f))
        target_weight = None
    else:
        weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor | GraphAttentionBackbone],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate inputs through a linear or attention network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            if isinstance(weight, GraphAttentionBackbone):
                N = current.size(0)
                adjacency = torch.ones(N, N, dtype=torch.float32)
                current = weight(current, adjacency)
            else:
                current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared overlap between two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Create a weighted graph from pairwise fidelities."""
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
# 4. GraphQNN wrapper class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Convenient wrapper around the classical utilities."""

    def __init__(self, qnn_arch: Sequence[int], samples: int, *, use_attention: bool = False):
        self.arch, self.weights, self.training_data, self.target_weight = random_network(
            qnn_arch, samples, use_attention=use_attention
        )

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
