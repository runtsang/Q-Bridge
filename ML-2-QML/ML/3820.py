"""Classical hybrid graph‑quantum neural network utilities."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# Utility functions (from GraphQNN.py with minor enhancements)
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs using the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and training data for its final layer."""
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
    """Run a forward pass through the classical feed‑forward network."""
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
    """Squared overlap between two normalized classical feature vectors."""
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
    """Construct a graph from pairwise classical fidelities."""
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
# Hybrid classical model combining a CNN encoder and a graph neural network
# --------------------------------------------------------------------------- #

class GraphQNNHybrid(nn.Module):
    """
    Classical hybrid model that first encodes 2‑D inputs with a CNN
    (inspired by Quantum‑NAT) and then passes the flattened features
    through a graph‑structured neural network.

    The architecture is configurable through ``cnn_cfg`` and ``gcn_cfg``.
    """

    def __init__(self, cnn_cfg: dict | None = None, gcn_cfg: dict | None = None):
        super().__init__()
        # Default CNN configuration mirrors QFCModel from Quantum‑NAT
        cnn_cfg = cnn_cfg or {
            "in_channels": 1,
            "features": [8, 16],
            "kernel_size": 3,
            "padding": 1,
            "pool_size": 2,
        }
        self.encoder = nn.Sequential(
            nn.Conv2d(cnn_cfg["in_channels"], cnn_cfg["features"][0], kernel_size=cnn_cfg["kernel_size"],
                      stride=1, padding=cnn_cfg["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(cnn_cfg["pool_size"]),
            nn.Conv2d(cnn_cfg["features"][0], cnn_cfg["features"][1], kernel_size=cnn_cfg["kernel_size"],
                      stride=1, padding=cnn_cfg["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(cnn_cfg["pool_size"]),
        )
        # Compute the size after pooling
        dummy = torch.zeros(1, cnn_cfg["in_channels"], 28, 28)  # MNIST‑style
        out = self.encoder(dummy)
        flatten_dim = out.view(1, -1).size(1)

        # Graph neural network configuration
        gcn_cfg = gcn_cfg or {"layers": [flatten_dim, 64, 4]}
        self.gnn_weights = nn.ParameterList()
        for in_f, out_f in zip(gcn_cfg["layers"][:-1], gcn_cfg["layers"][1:]):
            self.gnn_weights.append(nn.Parameter(_random_linear(in_f, out_f)))
        self.norm = nn.BatchNorm1d(gcn_cfg["layers"][-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: CNN → flatten → graph neural network → batch norm."""
        # Encode image
        encoded = self.encoder(x)
        flattened = encoded.view(x.size(0), -1)
        # Feed through GNN
        current = flattened
        activations = [flattened]
        for w in self.gnn_weights:
            current = torch.tanh(w @ current.t()).t()
            activations.append(current)
        out = self.norm(activations[-1])
        return out

    # --------------------------------------------------------------------- #
    # Utility wrappers that expose the same interface as the original GraphQNN
    # --------------------------------------------------------------------- #
    def generate_random_network(self, samples: int = 10) -> tuple:
        """Return architecture, weights, training data, and target weight."""
        arch = [w.shape[1] for w in self.gnn_weights] + [self.gnn_weights[-1].shape[0]]
        weights = [w.clone() for w in self.gnn_weights]
        target = weights[-1]
        train = random_training_data(target, samples)
        return arch, weights, train, target

    def compute_fidelity_graph(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> nx.Graph:
        """Compute the adjacency graph from the activations of the network."""
        activations = feedforward([w.shape[1] for w in self.gnn_weights] + [self.gnn_weights[-1].shape[0]],
                                  [w for w in self.gnn_weights], samples)
        # Use the final layer activations
        final_states = [act[-1] for act in activations]
        return fidelity_adjacency(final_states, threshold=0.8)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNHybrid",
]
