"""Hybrid classical graph neural network module.

This module implements a graph neural network that can be used purely
classically or as a building block for a quantum‑enhanced pipeline.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """Classical graph neural network with optional hybrid support.

    Parameters
    ----------
    arch : Sequence[int]
        List of node feature dimensionalities for each layer. The first
        element is the input dimensionality.
    n_qubits : int, default 0
        Number of quantum wires to allocate when the model operates in
        quantum mode.  For the pure classical implementation this
        parameter is ignored.
    """
    def __init__(self, arch: Sequence[int], n_qubits: int = 0) -> None:
        super().__init__()
        self.arch = list(arch)
        self.n_qubits = n_qubits
        self.linears = nn.ModuleList(
            [nn.Linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1)]
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape ``(num_nodes, in_features)``.

        Returns
        -------
        List[Tensor]
            List of activations for each layer including the input.
        """
        activations: List[Tensor] = [x]
        current = x
        for linear in self.linears:
            current = torch.tanh(linear(current))
            activations.append(current)
        return activations

    # ------------------------------------------------------------------
    #  Static helpers – these mirror the original seed implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random network and a matching training set.

        Parameters
        ----------
        qnn_arch : Sequence[int]
            Architecture of the network.
        samples : int
            Number of training samples.

        Returns
        -------
        Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]
            Architecture, list of random weight tensors, training data,
            and the target weight (final layer).
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNN._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two normalized vectors."""
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
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNN",
]
