import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Iterable, Sequence

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create a training set for a linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


class GraphQNNGenML:
    """Classical graph‑neural‑network generator.

    The API mirrors :class:`GraphQNNGenQML` to enable side‑by‑side
    classical‑vs‑quantum experiments.  It supports random network
    construction, feed‑forward propagation, fidelity‑based adjacency
    graphs, and a lightweight sampler module.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.weights: List[Tensor] = []
        self._build_random_weights()

    def _build_random_weights(self) -> None:
        self.weights = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Return architecture, weights, training data and target weight."""
        target_weight = self.weights[-1]
        training_data = random_training_data(target_weight, samples)
        return self.arch, self.weights, training_data, target_weight

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in self.weights:
                current = torch.tanh(w @ current)
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
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGenML.state_fidelity(si, sj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


class SamplerQNN(nn.Module):
    """A simple softmax sampler used in classical experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


__all__ = [
    "GraphQNNGenML",
    "SamplerQNN",
]
