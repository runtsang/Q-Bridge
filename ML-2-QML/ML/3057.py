"""GraphQNNGen330: hybrid graph neural network interface.

The class offers a unified API for both classical and quantum
implementations.  Classical operations are built on PyTorch,
while the quantum branch is handled by the companion QML module.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor
DataSample = Tuple[Tensor, Tensor]
NetworkArch = Sequence[int]
WeightList = List[Tensor]


class GraphQNNGen330:
    """
    Hybrid graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths, the first entry is the input dimension.
    is_quantum : bool, default False
        Flag indicating whether the instance should behave as a quantum
        network.  When True the methods delegate to the QML module.
    """

    def __init__(self, arch: NetworkArch, is_quantum: bool = False) -> None:
        self.arch = list(arch)
        self.is_quantum = is_quantum

    # ------------------------------------------------------------------
    # Classical helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[DataSample]:
        dataset: List[DataSample] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def _build_classical_network(self) -> None:
        layers: List[nn.Module] = []
        in_dim = self.arch[0]
        for out_dim in self.arch[1:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.arch[-1]))
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def random_network(
        cls,
        arch: NetworkArch,
        samples: int,
        is_quantum: bool = False,
    ) -> Tuple[NetworkArch, Union[WeightList, List[List[Tensor]]], List, Union[Tensor, Tensor]]:
        """
        Generate a random network and a matching training set.

        For the classical branch the last layer’s weights are used as the
        target; for the quantum branch a random unitary is drawn
        (handled by the QML module).
        """
        if is_quantum:
            raise NotImplementedError(
                "Quantum random_network is provided in the QML module."
            )
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(cls._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = cls.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    def feedforward(
        self,
        samples: Iterable[DataSample],
    ) -> List[List[Tensor]]:
        """
        Run a forward pass over *samples* and return a list of
        activations per sample.  The implementation is
        dispatch‑based: classical samples are processed by the
        PyTorch network; quantum samples would be processed
        by the QML module (not shown here).
        """
        if self.is_quantum:
            raise NotImplementedError(
                "Quantum feedforward is provided in the QML module."
            )
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.network:
                current = layer(current)
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------
    # Fidelity utilities – identical for both regimes
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return squared overlap between two pure state vectors."""
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
        """Construct a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = GraphQNNGen330.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Classifier construction – mirrors the quantum helper
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Return a feed‑forward classifier identical in structure to the
        quantum version.  The return tuple matches the quantum helper
        signature: (network, encoding, weight_sizes, observables).
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["GraphQNNGen330"]
