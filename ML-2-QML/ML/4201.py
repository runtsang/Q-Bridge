"""Combined classical GraphQNNHybrid module integrating GraphQNN utilities, regression dataset, and sampler.

This module provides:
- Random graph neural network generation with classical weights.
- Feedforward propagation for classical networks.
- Fidelity-based adjacency graph construction.
- Superposition data generation for regression tasks.
- Classical regression dataset and model.
- Classical sampler neural network.
"""

import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Sequence, Tuple, Any

Tensor = torch.Tensor

# --- Utility functions (mirror GraphQNN.py) ---------------------------------

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
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Superposition dataset (QuantumRegression.py) ----------------------------

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --- Classical regression model --------------------------------------------

class QModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: Tensor) -> Tensor:
        return self.net(state_batch).squeeze(-1)

# --- Classical sampler network ---------------------------------------------

def SamplerQNN() -> nn.Module:
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: Tensor) -> Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

# --- Unified class ---------------------------------------------------------

class GraphQNNHybrid:
    feedforward = staticmethod(feedforward)
    fidelity_adjacency = staticmethod(fidelity_adjacency)
    random_network = staticmethod(random_network)
    random_training_data = staticmethod(random_training_data)
    state_fidelity = staticmethod(state_fidelity)
    generate_superposition_data = staticmethod(generate_superposition_data)
    RegressionDataset = RegressionDataset
    QModel = QModel
    SamplerQNN = SamplerQNN

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "SamplerQNN",
    "GraphQNNHybrid",
]
