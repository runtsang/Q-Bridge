"""Combined classical graph neural network with estimator support.

This module extends the original GraphQNN utilities by exposing a
``GraphQNNGen112`` class that can be instantiated as a PyTorch
``nn.Module``.  The network is fully parameterised by an integer
architecture list and can be used either for pure feed‑forward
propagation or as a toy estimator via the ``EstimatorQNN`` helper
from the original example.  Additional static helpers generate random
weights, training data and a fidelity‑based adjacency graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


class GraphQNNGen112(nn.Module):
    """A PyTorch implementation of a graph‑neural‑network style feed‑forward net.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths, e.g. ``[2, 8, 4, 1]``.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        # Drop the final activation for a regressor
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> List[Tensor]:
        """Return activations at every layer (including input)."""
        activations: List[Tensor] = [inputs]
        x = inputs
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared overlap between two state vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = GraphQNNGen112.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic training data for a given linear layer."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random network and associated training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            weights.append(w)
        target_weight = weights[-1]
        training_data = GraphQNNGen112.random_training_data(
            target_weight, samples
        )
        return arch, weights, training_data, target_weight


def EstimatorQNN() -> nn.Module:
    """Return a simple fully‑connected regression network.

    Mirrors the original EstimatorQNN example but returns a PyTorch
    module that can be trained with standard optimiser routines.
    """
    return GraphQNNGen112([2, 8, 4, 1])


__all__ = [
    "GraphQNNGen112",
    "EstimatorQNN",
]
