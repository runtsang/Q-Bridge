"""Graph‑based neural network utilities with hybrid loss and dataset generation.

This module extends the original GraphQNN by adding:
* A two‑layer MLP head that maps the final hidden state to a real‑valued target.
* A `HybridLoss` that blends mean‑square error with a fidelity penalty.
* A `generate_graph_dataset` helper that builds random graphs and feeds them through the network.
* A `full_fidelity_matrix` helper that returns the full fidelity matrix for a set of states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
    *,
    device: str | None = None,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for training a single‑layer linear regression on the target weight."""
    device = device or "cpu"
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32, device=device)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and a synthetic training set."""
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
    """Run a forward pass through the linear network and record activations."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_outputs = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two normalized vectors."""
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
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def full_fidelity_matrix(states: Sequence[Tensor]) -> torch.Tensor:
    """Return the full fidelity matrix for a list of state vectors."""
    n = len(states)
    mat = torch.empty(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(i, n):
            fid = state_fidelity(states[i], states[j])
            mat[i, j] = fid
            mat[j, i] = fid
    return mat


def generate_graph_dataset(
    num_graphs: int,
    qnn_arch: Sequence[int],
    samples_per_graph: int,
) -> List[Dict]:
    """Generate a list of random graphs with associated training data.

    Each dictionary contains:
        - 'graph': a networkx.Graph instance
        - 'activations': activations for each sample
        -'states': the final hidden states for each sample
    """
    dataset: List[Dict] = []
    for _ in range(num_graphs):
        # Random graph
        g = nx.gnp_random_graph(num_nodes=qnn_arch[0], p=0.3)
        # Random network
        arch, weights, training_data, _ = random_network(qnn_arch, samples_per_graph)
        # Forward pass
        activations = feedforward(arch, weights, training_data)
        states = [act[-1] for act in activations]
        dataset.append({"graph": g, "activations": activations, "states": states})
    return dataset


class HybridLoss(nn.Module):
    """Blend MSE loss with a fidelity penalty between predictions and true targets.

    The fidelity term encourages predictions to lie close to the target in Hilbert‑space sense.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse_loss = self.mse(pred, target)
        # Compute fidelity between normalized predictions and targets
        pred_norm = pred / (torch.norm(pred, dim=0, keepdim=True) + 1e-12)
        target_norm = target / (torch.norm(target, dim=0, keepdim=True) + 1e-12)
        fid = torch.clamp(torch.abs(torch.dot(pred_norm, target_norm)), 0.0, 1.0)
        fid_loss = 1.0 - fid
        return self.alpha * mse_loss + (1.0 - self.alpha) * fid_loss


class SimpleMLPHead(nn.Module):
    """Two‑layer MLP that maps the final hidden state to a real‑valued target."""

    def __init__(self, in_features: int, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.tanh(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class GraphQNNHybridModel(nn.Module):
    """Full hybrid model that runs a linear network followed by a classical head."""

    def __init__(self, qnn_arch: Sequence[int], head_hidden: int = 16):
        super().__init__()
        self.arch = list(qnn_arch)
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.head = SimpleMLPHead(qnn_arch[-1], hidden=head_hidden)

    def forward(self, x: Tensor) -> Tensor:
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
        return self.head(current)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "full_fidelity_matrix",
    "generate_graph_dataset",
    "HybridLoss",
    "SimpleMLPHead",
    "GraphQNNHybridModel",
]
