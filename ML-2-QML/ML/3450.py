from __future__ import annotations

from typing import List, Sequence, Tuple, Iterable, Optional
import itertools
import networkx as nx
import torch

Tensor = torch.Tensor

class GraphQNNGen065:
    """Classical graph neural network that mirrors the quantum interface.

    The API matches the quantum counterpart: ``feedforward``, ``state_fidelity``,
    ``fidelity_adjacency``, and a ``sampler_qnn`` helper.  It can be used
    for classical simulations, synthetic data generation, or as a baseline
    for hybrid studies.
    """

    def __init__(self, architecture: Sequence[int]) -> None:
        """Create a random network with the given layer sizes."""
        self.arch = list(architecture)
        self.weights: List[Tensor] = [
            self._random_linear(in_, out) for in_, out in zip(self.arch[:-1], self.arch[1:])
        ]

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @classmethod
    def random_network(cls, architecture: Sequence[int], samples: int):
        """Generate a random network and a target training set.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        weights = [
            cls._random_linear(in_, out) for in_, out in zip(architecture[:-1], architecture[1:])
        ]
        target_weight = weights[-1]
        training_data = cls.random_training_data(target_weight, samples)
        return list(architecture), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Create synthetic input/target pairs based on a linear map."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass for a batch of input/target pairs.

        Parameters
        ----------
        samples:
            Iterable of (input, target) tuples. The target is ignored.
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
            current = features
            for w in self.weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two unit‑norm vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen065.state_fidelity(sa, sb)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ----------------------------------------------------------------------
    # SamplerQNN integration
    # ----------------------------------------------------------------------
    def sampler_qnn(self) -> torch.nn.Module:
        """Return a small softmax sampler network.

        Mirrors the SamplerQNN class from the reference project but
        is instantiated inside the GraphQNNGen065 context for
        end‑to‑end usage.
        """
        class SamplerModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(2, 4),
                    torch.nn.Tanh(),
                    torch.nn.Linear(4, 2),
                )

            def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
                return torch.nn.functional.softmax(self.net(inputs), dim=-1)

        return SamplerModule()

__all__ = [
    "GraphQNNGen065",
]
