"""GraphQNN: a classical MLP with graph‑aware training utilities.

This module augments the original seed by adding a differentiable training pipeline,
adaptive learning‑rate, and a graph‑based similarity metric for model comparison.
The API mirrors the quantum version so that a user can swap implementations without
changing the rest of the code base.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.optim import Adam

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Core utilities (unchanged from the seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=False)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset where the target is a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random network and a synthetic training set."""
    # Build a fresh model to obtain a target weight
    model = GraphQNN(qnn_arch)
    for param in model.parameters():
        torch.nn.init.normal_(param)
    target_weight = model.layers[-1].weight.clone().detach()
    training_data = random_training_data(target_weight, samples)
    # Extract initial weights
    init_weights = [p.detach().clone() for p in model.parameters()]
    return list(qnn_arch), init_weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# GraphQNN model definition
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """A simple feed‑forward neural network with tanh activations."""

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> List[Tensor]:
        activations = [x]
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

# --------------------------------------------------------------------------- #
# Forward propagation helper (seed‑compatible)
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the network on a batch of samples, returning activations per sample."""
    model = GraphQNN(qnn_arch)
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(w)
        results: List[List[Tensor]] = []
        for x, _ in samples:
            activations = model(x)
            results.append(activations)
    return results

# --------------------------------------------------------------------------- #
# Training, prediction and evaluation helpers
# --------------------------------------------------------------------------- #
def train(
    qnn_arch: Sequence[int],
    init_weights: Sequence[Tensor],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    target_weight: Tensor,
    lr: float = 1e-3,
    epochs: int = 200,
) -> List[Tensor]:
    """Fit a GraphQNN so that its last‑layer weight matches `target_weight`."""
    model = GraphQNN(qnn_arch)
    # initialise parameters
    with torch.no_grad():
        for param, w in zip(model.parameters(), init_weights):
            param.copy_(w)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        epoch_loss = 0.0
        for x, y in training_data:
            optimizer.zero_grad()
            out = model(x)[-1]
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(training_data)
    trained_weights = [p.detach().clone() for p in model.parameters()]
    return trained_weights

def predict(qnn_arch: Sequence[int], weights: Sequence[Tensor], x: Tensor) -> Tensor:
    """Return the final layer output for a single input `x`."""
    model = GraphQNN(qnn_arch)
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(w)
        return model(x)[-1]

def evaluate(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    target_weight: Tensor,
) -> float:
    """Return the mean fidelity of the model output against the target weight."""
    model = GraphQNN(qnn_arch)
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(w)
        total_fid = 0.0
        for x, y in training_data:
            out = model(x)[-1]
            fid = state_fidelity(out, y)
            total_fid += fid
    return total_fid / len(training_data)

# --------------------------------------------------------------------------- #
# Graph‑based similarity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two tensors treated as pure states."""
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
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
    "predict",
    "evaluate",
]
