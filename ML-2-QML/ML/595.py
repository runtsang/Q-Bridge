"""GraphQNN__gen244 - Classical baseline with hybrid extensions.

This module extends the original seed by adding (i) batched feed‑forward that
supports PyTorch tensors, (ii) a simple self‑attention layer that uses the
graph built from state fidelities, and (iii) an adaptive learning‑rate
optimizer.  The same API can be dropped into a quantum pipeline that
uses the same adjacency graph.  The code is fully importable and can
be used in the same way as the original GraphQNN module.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int, seed: int | None = None) -> Tensor:
    """Return a random weight matrix with optional reproducible seed."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> list[tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, y) where y = weight @ x."""
    dataset: list[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int, seed: int | None = None) -> tuple[list[int], list[Tensor], list[tuple[Tensor, Tensor]], Tensor]:
    """Create a random multilayer perceptron and synthetic training data."""
    weights: list[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f, seed))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[tuple[Tensor, Tensor]]) -> list[list[Tensor]]:
    """Classic feed‑forward with tanh activations, returns activations per layer."""
    stored: list[list[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two real vectors (normalized)."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


@dataclass
class GraphQNN__gen244:
    """Hybrid classical‑quantum graph neural network base."""
    arch: Sequence[int]
    weights: list[Tensor] | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.device = self.device or torch.device("cpu")
        if self.weights is None:
            self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        # Convert to parameters for optimizer support
        self.weights = [torch.nn.Parameter(w) for w in self.weights]

    def forward_batch(self, inputs: Tensor) -> Tensor:
        """Batch feed‑forward returning final activations."""
        x = inputs.to(self.device)
        for w in self.weights:
            x = torch.tanh(w @ x)
        return x

    def self_attention(self, activations: list[Tensor], adjacency: nx.Graph) -> Tensor:
        """Simple graph‑based self‑attention over layer activations."""
        flat = torch.cat(activations, dim=0)
        # Build attention weights from adjacency degrees
        attn_weights = torch.tensor(
            [sum(adjacency[neighbor][v]["weight"] for neighbor in adjacency[v]) for v in range(adjacency.number_of_nodes())],
            dtype=torch.float32,
            device=self.device,
        )
        attn_weights = F.softmax(attn_weights, dim=0)
        return flat * attn_weights.to(flat.device)

    def train_step(self, data: list[tuple[Tensor, Tensor]], lr: float = 1e-3) -> float:
        """One SGD step with adaptive learning rate (AdamW)."""
        optimizer = torch.optim.AdamW(self.weights, lr=lr)
        loss_fn = nn.MSELoss()
        optimizer.zero_grad()
        loss = 0.0
        for x, y in data:
            pred = self.forward_batch(x.unsqueeze(0)).squeeze(0)
            loss += loss_fn(pred, y)
        loss /= len(data)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, data: list[tuple[Tensor, Tensor]], epochs: int = 10, lr: float = 1e-3) -> list[float]:
        losses: list[float] = []
        for _ in range(epochs):
            loss = self.train_step(data, lr)
            losses.append(loss)
        return losses


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen244",
]
