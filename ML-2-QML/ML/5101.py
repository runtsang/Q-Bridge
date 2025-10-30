"""Hybrid classical estimator that augments a feed‑forward network with a
graph‑based aggregation of intermediate activations.  The graph is
constructed from the pairwise fidelity of the hidden states, encouraging
the model to exploit structural relations between samples."""
from __future__ import annotations

import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Graph utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
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
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        current = features
        layer_states = [current]
        for w in weights:
            current = torch.tanh(w @ current)
            layer_states.append(current)
        activations.append(layer_states)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """Hybrid classical estimator that augments a feed‑forward network with a
    graph‑based aggregation of intermediate activations.  The graph is
    constructed from the pairwise fidelity of the hidden states, which
    encourages the model to exploit structural relations between samples.
    """

    def __init__(self, arch: Sequence[int] = (2, 8, 4, 1), threshold: float = 0.9) -> None:
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if out_f!= self.arch[-1]:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the classical network
        hidden: List[Tensor] = []
        current = x
        for layer in self.net:
            current = layer(current)
            if isinstance(layer, nn.Linear):
                hidden.append(current)
        # hidden[-1] corresponds to the last hidden layer before output
        outputs = hidden[-1]
        batch = outputs.shape[0]
        # Build adjacency from pairwise fidelity of the last hidden layer
        adj = torch.zeros((batch, batch), dtype=outputs.dtype, device=outputs.device)
        for i in range(batch):
            for j in range(i + 1, batch):
                fid = state_fidelity(outputs[i], outputs[j])
                if fid >= self.threshold:
                    adj[i, j] = adj[j, i] = 1.0
        # Graph‑aggregated representation
        aggregated = torch.matmul(adj, outputs)
        return aggregated

    def generate_random_dataset(self, samples: int = 100) -> Tuple[List[Tuple[Tensor, Tensor]], nx.Graph]:
        """Generate a synthetic dataset using a random network, along with the
        fidelity‑based graph of the latent activations.  Useful for quick
        sanity checks or pre‑training experiments.
        """
        _, weights, training_data, _ = random_network(self.arch, samples)
        activations = feedforward(self.arch, weights, training_data)
        last_states = [acts[-2] for acts in activations]
        graph = fidelity_adjacency(last_states, self.threshold)
        return training_data, graph


__all__ = [
    "EstimatorQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
