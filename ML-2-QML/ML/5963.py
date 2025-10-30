"""Graph‑based neural network with hybrid classical support.

This module implements a GraphQNNHybrid class that combines the
random‑network generation, fidelity‑based graph construction and a
hybrid dense head (either a linear layer or a quantum expectation
layer).  The public API mirrors the original GraphQNN.py so that
downstream code can remain unchanged.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Utility functions – random generation & fidelity helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random torch tensor of shape (out, input)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: torch.Tensor,
    samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate training samples for a target weight matrix."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Propagate samples through the classical network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Absolute squared overlap between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
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
#  Hybrid head – classical dense layer mimicking a quantum expectation
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Dense head that emulates a quantum expectation head using a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)

# --------------------------------------------------------------------------- #
#  GraphQNNHybrid – main model
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """
    Graph‑based neural network that can run purely classically or with a
    hybrid quantum head.  The architecture is defined by a list of layer
    sizes.  The forward pass returns a list of activation tensors and a
    binary probability distribution.
    """
    def __init__(self, arch: Sequence[int], use_quantum_head: bool = False, **head_kwargs):
        super().__init__()
        self.arch = list(arch)
        self.weights = nn.ParameterList(
            nn.Parameter(_random_linear(in_f, out_f))
            for in_f, out_f in zip(arch[:-1], arch[1:])
        )
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            # The quantum head must be attached later via `set_quantum_head`
            self.head = None
        else:
            self.head = HybridHead(arch[-1], **head_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        if self.use_quantum_head:
            if self.head is None:
                raise RuntimeError("Quantum head not attached – call `set_quantum_head`.")
            logits = self.head(current)
        else:
            logits = self.head(current)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return activations, probs

    def set_quantum_head(self, quantum_head: Any):
        """Attach a quantum hybrid head that implements a __call__ API."""
        self.head = quantum_head
        self.use_quantum_head = True

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "HybridHead",
    "GraphQNNHybrid",
]
