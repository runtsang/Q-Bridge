"""GraphQNN: Classical implementation with optional hybrid layers.

Features
--------
* Classical GNN stack using ``torch.nn.Linear`` layers.
* Optional hybrid quantum layer using the lightweight ``FCL`` wrapper.
* Random network generation (weights or parameterized circuits).
* Feed‑forward propagation and fidelity‑based adjacency graph construction.
* Estimation via ``FastBaseEstimator`` for batch evaluation.

The module stays fully importable and compatible with the original anchor
``GraphQNN.py`` while adding a hybrid extension inspired by Quanvolution,
FastBaseEstimator, and FCL.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn.functional import tanh

from.FCL import FCL
from.FastBaseEstimator import FastBaseEstimator


Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate synthetic (feature, target) pairs for a linear transformation."""
    dataset: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


class GraphQNNClassifier(nn.Module):
    """Classical graph neural network with optional hybrid quantum layers.

    Parameters
    ----------
    qnn_arch : list[int]
        Architecture defining the number of units per layer.
    use_quantum : bool, optional
        If ``True`` the final layer is replaced by a quantum fully‑connected
        layer (``FCL``).  The rest of the network remains classical.
    """

    def __init__(self, qnn_arch: Sequence[int], use_quantum: bool = False) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.use_quantum = use_quantum
        self.layers: nn.ModuleList = nn.ModuleList()

        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            if use_quantum and out_f == self.arch[-1]:
                # Final layer is quantum
                self.layers.append(FCL())
            else:
                self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> Tensor:
        """Propagate a batch of node features through the network."""
        activations: List[Tensor] = [x]
        current = x
        for layer in self.layers:
            if isinstance(layer, FCL):
                current = layer.run(current.detach().numpy())
                current = torch.as_tensor(current, dtype=torch.float32)
            else:
                current = tanh(layer(current))
            activations.append(current)
        return activations[-1]

    def random_weights(self, samples: int) -> None:
        """In‑place randomisation of all classical weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph from cosine similarities of node states."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            norm_i, norm_j = torch.norm(s_i), torch.norm(s_j)
            if norm_i.item() == 0 or norm_j.item() == 0:
                continue
            fid = float((s_i @ s_j) / (norm_i * norm_j))
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def evaluate(
        self,
        observables: Iterable[callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model with a batch of parameter sets via FastBaseEstimator."""
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "GraphQNNClassifier",
    "random_training_data",
    "fidelity_adjacency",
]
