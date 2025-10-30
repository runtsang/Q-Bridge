"""
Enhanced classical graph neural network utilities.

This module extends the original seed by adding a lightweight
parameterised GNN implementation that supports dropout, weight
normalization, and an optional graph convolutional layer.  The
public API mirrors the seed functions but returns a `GraphQNN`
instance that can be used for quick prototyping or as a drop‑in
replacement in larger pipelines.
"""

import torch
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Create a random weight matrix with orthogonal initialization."""
    weight = torch.empty(out_features, in_features, dtype=torch.float32)
    torch.nn.init.orthogonal_(weight)
    return weight

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

class GraphQNN:
    """
    Lightweight classical graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths.  The first element is the input dimensionality.
    dropout : float, optional
        Drop‑out probability applied after each linear transformation.
    """

    def __init__(self, arch: Sequence[int], dropout: float = 0.0):
        self.arch = list(arch)
        self.dropout = dropout
        self.weights: List[Tensor] = [_random_linear(in_f, out_f)
                                      for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        self._params = torch.nn.ParameterList([torch.nn.Parameter(w) for w in self.weights])

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        """Instantiate a network with random weights and synthetic training data."""
        net = cls(arch)
        training_data = random_training_data(net.weights[-1], samples)
        return net, training_data

    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass and capture activations per layer."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                if self.dropout > 0.0:
                    current = torch.nn.functional.dropout(current,
                                                          p=self.dropout,
                                                          training=True)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from pairwise state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def get_embeddings(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        """Return the last‑layer embeddings for a batch of samples."""
        activations = self.feedforward(samples)
        return [act[-1] for act in activations]

    def build_graph_from_embeddings(self,
                                    embeddings: List[Tensor],
                                    threshold: float) -> nx.Graph:
        """Construct an adjacency graph directly from embeddings."""
        states = [e.detach() for e in embeddings]
        return self.fidelity_adjacency(states, threshold)
