import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Randomly initialise a linear layer weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input, target) pairs using the supplied weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network mirroring GraphQNN."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate inputs through a series of linear layers with tanh activations."""
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
    """Cosine‑squared similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQuantumNAT(nn.Module):
    """
    Classical graph‑based neural network that emulates the Quantum‑NAT encoder
    followed by a graph‑structured feed‑forward path.
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)

        # Convolutional encoder identical to Quantum‑NAT
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Linear layers forming the graph‑like propagation
        self.fc_layers = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.fc_layers.append(nn.Linear(in_f, out_f))

        self.norm = nn.BatchNorm1d(self.qnn_arch[-1])

    def forward(self, x: Tensor) -> Tuple[Tensor, List[List[Tensor]]]:
        """
        Forward pass that returns the final output and the full list of
        activations at each layer (including the encoder output).
        """
        # Encoder
        features = self.encoder(x)  # shape: (bsz, 16, 7, 7)
        flattened = features.view(features.size(0), -1)  # (bsz, 16*7*7)

        activations = [flattened]
        current = flattened
        for layer in self.fc_layers:
            current = torch.tanh(layer(current))
            activations.append(current)

        out = self.norm(current)
        return out, activations

    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Convenience wrapper delegating to the module‑level helper."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )


__all__ = [
    "GraphQuantumNAT",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
