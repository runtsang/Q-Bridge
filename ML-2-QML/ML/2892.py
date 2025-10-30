"""Classical Graph Neural Network utilities with fidelity‑based graph construction.

This module re‑implements the core logic from the original GraphQNN seed while
adding a lightweight API for random network generation and state‑fidelity
metrics.  The design mirrors the quantum counterpart so that the same class
name can be swapped between classical and quantum back‑ends without
modifying downstream code."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


def _rand_lin(in_f: int, out_f: int) -> Tensor:
    """Return a random weight matrix of shape (out_f, in_f)."""
    return torch.randn(out_f, in_f, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs where y = weight @ x."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        data.append((x, weight @ x))
    return data


def random_network(arch: Sequence[int], samples: int):
    """Create a random linear network and associated training data."""
    weights: List[Tensor] = [_rand_lin(i, o) for i, o in zip(arch[:-1], arch[1:])]
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(arch), weights, training, target


def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Store activations for each sample through the linear network."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_outs = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            layer_outs.append(cur)
        activations.append(layer_outs)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
