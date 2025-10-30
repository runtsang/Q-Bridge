"""
GraphQNN: Classical MLP with supervised training and fidelity‑based graph construction.

Features
--------
* Random network generation mirroring the original seed.
* Feed‑forward propagation through the network.
* Mean‑squared‑error training loop with early stopping.
* Node‑label construction from flattened activations.
* Utility to build a weighted adjacency graph that stores node labels.

The public API matches the original seed but adds `train_network` and
`build_labeled_graph`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of (input, target) pairs for the given target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return (architecture, weight list, training data, target weight)."""
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
    """Return a list of activation lists for each sample."""
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
    """Return the squared cosine similarity between two vectors."""
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


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Simple feed‑forward network matching a given architecture."""

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        layers = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        # Remove the last activation
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_network(
    arch: Sequence[int],
    training_data: List[Tuple[Tensor, Tensor]],
    val_data: List[Tuple[Tensor, Tensor]],
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Train an MLP to predict the target weight from the input.
    Returns the trained weight matrix of the final layer and the list of all weights.
    """
    device = torch.device("cpu")
    net = MLP(arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    counter = 0
    best_state = None

    for epoch in range(epochs):
        net.train()
        for x, y in training_data:
            optimizer.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_data:
                pred = net(x)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(val_data)

        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            best_state = net.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break

    # Load best model
    net.load_state_dict(best_state)
    # Extract weights
    weights = []
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.clone())
    return weights[-1], weights


def flatten_activations(activations: List[List[Tensor]]) -> List[Tensor]:
    """Convert a list of activation lists into a flat feature vector per sample."""
    return [torch.cat([act.flatten() for act in sample]) for sample in activations]


def build_labeled_graph(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a fidelity graph and attach the flattened activation vector as a node
    attribute called ``features``.
    """
    graph = fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
    flat = flatten_activations([[s] for s in states])  # each state as its own sample
    for idx, feat in enumerate(flat):
        graph.nodes[idx]["features"] = feat
    return graph


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_network",
    "build_labeled_graph",
]
