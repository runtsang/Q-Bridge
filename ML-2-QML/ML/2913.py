"""Hybrid classical graph neural network with a regression head.

The class combines graph-based feature propagation (see GraphQNN) with a
fully‑connected regressor inspired by EstimatorQNN.  It exposes a
minimal training API that operates on synthetic data generated from a target
weight matrix.

The implementation deliberately mirrors the public API of the original
GraphQNN module so that downstream experiments can swap in the hybrid
variant without modification.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

class GraphEstimatorQNN(nn.Module):
    """
    A hybrid graph neural network + regressor.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the graph neural network.  The last element is the
        dimensionality of the output feature vector that will be fed into
        the regressor.
    regressor_arch : Sequence[int]
        Sizes of the hidden layers of the regression head.
    """

    def __init__(self, qnn_arch: Sequence[int], regressor_arch: Sequence[int] = (8, 4)):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.regressor_arch = list(regressor_arch)

        # Build the graph part
        self.graph_weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f))
             for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])]
        )

        # Build the regression head
        layers = []
        input_dim = self.qnn_arch[-1]
        for hidden in self.regressor_arch:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.Tanh())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through the graph network and the regression head."""
        current = features
        activations = [current]
        for weight in self.graph_weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        # regression
        output = self.regressor(current)
        return output, activations

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return the activations for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.graph_weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphEstimatorQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random instance of the hybrid network and synthetic data."""
        weights = [_random_linear(in_f, out_f)
                   for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

__all__ = [
    "GraphEstimatorQNN",
]
