"""UnifiedSamplerGraphQNN – classical component.

This module implements a hybrid architecture that:
* Wraps the original SamplerQNN neural network and extends it
  with graph‑based connectivity derived from the fidelity between
  states produced by the network.
* Uses the same layer‑wise activation pattern as the original
  SamplerQNN, but adds a new method `build_graph` that
  creates a *networkx* graph from the activations.
* Provides a helper `sample_from_graph` that runs the
  classical sampler on any node of the graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a linear map defined by `weight`."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random feed‑forward network and a matching training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight

def feedforward(qnn_arch: List[int], weights: List[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Compute layer‑wise activations for each sample."""
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
    """Return the squared cosine similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class UnifiedSamplerGraphQNN(nn.Module):
    """Classical sampler network with graph‑based fidelity adjacency."""

    def __init__(self, qnn_arch: List[int], graph_threshold: float = 0.9, secondary_threshold: float | None = None):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.weights = nn.ParameterList(
            nn.Parameter(_random_linear(in_f, out_f))
            for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the final softmax output of the sampler."""
        x = inputs
        for weight in self.weights:
            x = torch.tanh(weight @ x)
        return F.softmax(x, dim=-1)

    def build_graph(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> nx.Graph:
        """Construct a graph from the activations of the network."""
        activations = feedforward(self.qnn_arch, [w for w in self.weights], samples)
        flat_states = [act for sample in activations for act in sample]
        graph = fidelity_adjacency(flat_states, self.graph_threshold, secondary=self.secondary_threshold)
        # Attach the activation vector to each node
        for idx, state in enumerate(flat_states):
            graph.nodes[idx]["activation"] = state
        return graph

    def sample_from_graph(self, graph: nx.Graph, node: int) -> Tensor:
        """Return the activation at the specified graph node."""
        return graph.nodes[node]["activation"]

__all__ = [
    "UnifiedSamplerGraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
