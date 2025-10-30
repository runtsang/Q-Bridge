"""Enhanced classical Graph Neural Network module.

Features
--------
* Linear chain of layers with optional dropout and weight‑norm.
* `feedforward` returns activations per layer for each sample.
* `train_model` runs a mini‑batch gradient descent loop.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialised weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` input–output pairs from a fixed linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear chain of layers and a training set for the last layer."""
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
    """Return a list of activations for each sample."""
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
    """Cosine‑like fidelity between two unit‑norm vectors."""
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
    """Build a weighted graph where edges are selected by fidelity."""
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
# New additions – a simple neural‑network wrapper and training utilities
# --------------------------------------------------------------------------- #

class _LinearLayer(nn.Module):
    """Linear layer with optional weight‑norm and dropout."""
    def __init__(self, in_f: int, out_f: int, *, dropout: float = 0.0, weight_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=True)
        if weight_norm:
            nn.utils.weight_norm(self.linear)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.dropout(self.linear(x)))

class GraphNet(nn.Module):
    """A linear chain of layers that mimics the original architecture."""
    def __init__(self, architecture: Sequence[int], *, dropout: float = 0.0, weight_norm: bool = True):
        super().__init__()
        self.arch = list(architecture)
        self.layers = nn.ModuleList(
            [_LinearLayer(in_f, out_f, dropout=dropout, weight_norm=weight_norm)
             for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the final output."""
        for layer in self.layers:
            x = layer(x)
        return x

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for each sample."""
        activations = []
        for features, _ in samples:
            acts = [features]
            current = features
            for layer in self.layers:
                current = layer(current)
                acts.append(current)
            activations.append(acts)
        return activations

    def predict(self, x: Tensor) -> Tensor:
        """Return the final output for a single sample."""
        return self.forward(x)

    def train_model(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """Mini‑batch training loop."""
        self.train()
        data = list(samples)
        for epoch in range(epochs):
            torch.random.manual_seed(epoch)
            torch.random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                xs = torch.stack([x for x, _ in batch])
                ys = torch.stack([y for _, y in batch])
                optimizer.zero_grad()
                preds = self(xs)
                loss = loss_fn(preds, ys)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

__all__ = [
    "GraphNet",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
