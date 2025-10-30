"""Enhanced classical graph neural network with stochastic layer sampling and training support.

The module keeps the original API but extends it with:
* `sampled_feedforward`: runs a forward pass while randomly dropping layers.
* `train_one_epoch`: a simple Adam optimisation loop that updates the last weight matrix.
* `random_graph`: exposes a networkx graph of the architecture for visualisation.

All functions are pure and fully importable.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn, optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
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


def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def sampled_feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]], drop_prob: float = 0.2, seed: int | None = None) -> List[List[Tensor]]:
    rng = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            if torch.rand(1, generator=rng).item() < drop_prob:
                activations.append(current)
                continue
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def train_one_epoch(weights: List[Tensor], training_data: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, weight_decay: float = 0.0, loss_fn: nn.Module = nn.MSELoss()) -> float:
    target = weights[-1]
    optimizer = optim.Adam([target], lr=lr, weight_decay=weight_decay)
    epoch_loss = 0.0
    for features, target_y in training_data:
        optimizer.zero_grad()
        current = features
        for w in weights[:-1]:
            current = torch.tanh(w @ current)
        pred = torch.tanh(target @ current)
        loss = loss_fn(pred, target_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(training_data)


def random_graph(qnn_arch: Sequence[int]) -> nx.Graph:
    G = nx.DiGraph()
    G.add_nodes_from(range(len(qnn_arch)))
    for i in range(len(qnn_arch)-1):
        G.add_edge(i, i+1, weight=qnn_arch[i+1])
    return G


class GraphQNN__gen130:
    """Wrapper class exposing the original API along with new extensions."""

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def sampled_feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]], drop_prob: float = 0.2, seed: int | None = None) -> List[List[Tensor]]:
        return sampled_feedforward(qnn_arch, weights, samples, drop_prob=drop_prob, seed=seed)

    @staticmethod
    def train_one_epoch(weights: List[Tensor], training_data: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, weight_decay: float = 0.0, loss_fn: nn.Module = nn.MSELoss()) -> float:
        return train_one_epoch(weights, training_data, lr=lr, weight_decay=weight_decay, loss_fn=loss_fn)

    @staticmethod
    def random_graph(qnn_arch: Sequence[int]) -> nx.Graph:
        return random_graph(qnn_arch)


__all__ = [
    "GraphQNN__gen130",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "sampled_feedforward",
    "train_one_epoch",
    "random_graph",
]
