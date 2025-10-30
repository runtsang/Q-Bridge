"""Unified classical estimator that combines classical, graph, and quantum-derived features.

The module defines `UnifiedEstimatorQNN`, a `torch.nn.Module` that receives a batch of
features, optionally propagates them through a graph‑based QNN, extracts classical
features from the quantum circuit (if provided), and finally regresses the target with
a multi‑layer perceptron.  The design keeps each component isolated yet compatible,
enabling end‑to‑end experiments that juxtapose classical, quantum, and graph‑theoretic
insights.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# 1. Graph utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


def feedforward(qnn_arch: list[int], weights: list[torch.Tensor], samples: list[tuple[torch.Tensor, torch.Tensor]]):
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 2. Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 3. UnifiedEstimatorQNN
# --------------------------------------------------------------------------- #
class UnifiedEstimatorQNN(nn.Module):
    """Hybrid estimator that merges raw features, optional graph features, and optional quantum features."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int],
        qfeat_dim: int = 0,
        num_graph_layers: int = 0,
        graph_threshold: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_features = num_features

        # Feed‑forward head
        in_dim = num_features + 1 + qfeat_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers).to(self.device)

        # Graph parameters
        self.num_graph_layers = num_graph_layers
        self.graph_threshold = graph_threshold
        self.graph_adj = None if num_graph_layers == 0 else None

    @torch.no_grad()
    def _build_graph(self, states: torch.Tensor) -> torch.Tensor:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = state_fidelity(state_i, state_j)
            if fid >= self.graph_threshold:
                graph.add_edge(i, j, weight=1.0)
        adj = nx.to_numpy_array(graph)
        return torch.tensor(adj, dtype=torch.float32, device=self.device)

    def forward(
        self,
        x: torch.Tensor,
        quantum_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        batch = x.shape[0]

        # Graph features
        if self.num_graph_layers > 0:
            graph_adj = self._build_graph(x)
            graph_feat = graph_adj.mean(dim=1, keepdim=True)
        else:
            graph_feat = torch.zeros(batch, 1, device=self.device)

        # Concatenate features
        features = [x]
        if quantum_features is not None:
            features.append(quantum_features.to(self.device))
        features.append(graph_feat)
        all_features = torch.cat(features, dim=1)

        return self.head(all_features).squeeze(-1)

    @staticmethod
    def get_dataloader(samples: int, num_features: int, batch_size: int = 32):
        dataset = RegressionDataset(samples, num_features)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


__all__ = [
    "UnifiedEstimatorQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
