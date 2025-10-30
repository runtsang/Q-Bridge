"""Hybrid classical graph neural network implementation.

This module provides a GraphQNNHybrid class that offers a classical
feed‑forward network with optional graph‑based adjacency construction.
It is compatible with the original GraphQNN utilities while adding
support for sampling layers and hybrid training strategies.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


class GraphQNNHybrid:
    """Hybrid graph neural network that can be used in purely classical
    settings or as a bridge to quantum layers.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        """Create a network with the given layer sizes.

        Parameters
        ----------
        arch:
            Sequence of node counts per layer, e.g. ``[2, 4, 2]``.
        """
        self.arch = list(arch)

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    def random_weights(self) -> List[Tensor]:
        """Generate a list of random weight matrices for each layer."""
        return [
            self._random_linear(in_f, out_f)
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]

    def random_training_data(
        self, weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic input–output pairs for a target weight matrix."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(
        self,
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate each sample through the network.

        The activation function is ``tanh`` for all layers.
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared cosine similarity between two vectors."""
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
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Optional sampler integration
    # ------------------------------------------------------------------
    @staticmethod
    def SamplerQNN() -> torch.nn.Module:
        """Return a simple softmax sampler network."""
        class SamplerModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(2, 4),
                    torch.nn.Tanh(),
                    torch.nn.Linear(4, 2),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return torch.nn.functional.softmax(self.net(inputs), dim=-1)

        return SamplerModule()
