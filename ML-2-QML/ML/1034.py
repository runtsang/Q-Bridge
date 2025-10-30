"""Hybrid Graph Neural‑Quantum Network – Classical utilities.

This module extends the original GraphQNN utilities by adding a
classical GNN encoder that maps graph features to a quantum state
(amplitude vector) and a simple hybrid evaluation routine.  The
public API mirrors the seed but now includes a ``GraphEncoder`` class
and a ``hybrid_fidelity`` function that can be used in downstream
experiments.  All operations are implemented with NumPy/PyTorch and
are fully classical.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor
Array = np.ndarray


# --------------------------------------------------------------------------- #
# 1.  Helper utilities – same as the seed but with NumPy fallback
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate (x, y) pairs where y = W x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.shape[1], dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return a network architecture, weight list, training data and the target
    (last layer) weight matrix."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(qnn_arch), weights, training, target


# --------------------------------------------------------------------------- #
# 2.  Forward pass – same as the seed
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of per‑layer activations for each sample."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_vals = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


# --------------------------------------------------------------------------- #
# 3.  Fidelity utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
# 4.  Classical GNN encoder – maps graph to amplitude vector
# --------------------------------------------------------------------------- #
class GraphEncoder(nn.Module):
    """Simple 2‑layer GCN that maps a graph to a quantum state amplitude vector.

    The encoder outputs a vector of length ``2 ** out_qubits`` which can be
    interpreted as the amplitude of a pure state on ``out_qubits`` qubits.
    """

    def __init__(self, in_feats: int, hidden: int, out_qubits: int):
        super().__init__()
        self.hidden = hidden
        self.out_qubits = out_qubits
        self.linear1 = nn.Linear(in_feats, hidden)
        self.linear2 = nn.Linear(hidden, 2 ** out_qubits)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.linear1(x))
        out = self.linear2(h)
        # normalize to unit norm to represent a pure state
        return out / (torch.norm(out) + 1e-12)


def graph_to_amplitude(
    graph: nx.Graph, encoder: GraphEncoder
) -> Tensor:
    """Convert a graph into a quantum state amplitude vector using ``encoder``.

    The graph is represented by a simple degree‑based feature vector:
    the mean degree of all nodes is used as a one‑dimensional feature.
    """
    degrees = torch.tensor([d for _, d in graph.degree()], dtype=torch.float32)
    feat = degrees.mean().unsqueeze(0)  # shape (1,)
    return encoder(feat)


# --------------------------------------------------------------------------- #
# 5.  Hybrid fidelity – compares two amplitude vectors
# --------------------------------------------------------------------------- #
def hybrid_fidelity(pred_state: Tensor, target_state: Tensor) -> float:
    """Compute fidelity between two classical amplitude vectors."""
    return state_fidelity(pred_state, target_state)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphEncoder",
    "graph_to_amplitude",
    "hybrid_fidelity",
]
