"""Combined classical regression and graph utilities.

This module merges the regression dataset, feed‑forward network,
graph‑based fidelity adjacency, and an EstimatorQNN wrapper.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import networkx as nx
from typing import Iterable, List, Tuple, Sequence

# ----- Data generation ---------------------------------------------------------

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a toy regression problem where the target is a smooth
    function of a linear combination of the input features.
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Standard PyTorch dataset for the toy regression problem."""
    def __init__(self, samples: int, num_features: int):
        self.X, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ----- Graph utilities ---------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Create a random feed‑forward network and a synthetic training set."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Run a forward pass and record all hidden activations."""
    activations: List[List[torch.Tensor]] = []
    for x, _ in samples:
        layer_outputs = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph where edges encode state fidelity."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----- Classical regression model ---------------------------------------------

class RegressionModel(nn.Module):
    """A small neural network that mirrors the quantum architecture."""
    def __init__(self, num_features: int, hidden_sizes: Sequence[int] = (32, 16)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ----- EstimatorQNN wrapper -----------------------------------------------------

class EstimatorQNNModel(nn.Module):
    """Thin wrapper around a tiny feed‑forward regressor."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----- Classifier circuit factory ---------------------------------------------

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a feed‑forward classifier and metadata that mimics the quantum
    version: the returned iterables encode the positions of encoding
    and weight parameters, and the list of observable indices.
    """
    layers: List[nn.Module] = []
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(num_features, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())

    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    net = nn.Sequential(*layers)
    observables: List[int] = list(range(2))
    return net, encoding, weight_sizes, observables

# ------------------------------------------------------------------------------

__all__ = [
    "RegressionDataset",
    "RegressionModel",
    "EstimatorQNNModel",
    "build_classifier_circuit",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "generate_superposition_data",
]
