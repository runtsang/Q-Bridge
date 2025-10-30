from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility functions for classical graph neural network
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data from a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight network and synthetic training data."""
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
    """Forward pass through the network for a batch of samples."""
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
    """Squared overlap between two normalized vectors."""
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
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Hybrid Graph Neural Network with Classical Self‑Attention
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """
    A hybrid graph neural network that augments each linear layer with a
    classical self‑attention block.  The network is fully differentiable
    and can be trained with standard PyTorch optimisers.
    """

    def __init__(self, qnn_arch: Sequence[int], attention_dim: int = 4, use_self_attention: bool = True):
        self.qnn_arch = list(qnn_arch)
        self.use_self_attention = use_self_attention
        self.attention_dim = attention_dim

        # Initialise linear layers
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(_random_linear(in_f, out_f))
                for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
            ]
        )

        # Initialise self‑attention parameters per hidden layer
        if use_self_attention:
            self.rotation_params = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(attention_dim, attention_dim)) for _ in range(len(qnn_arch) - 1)]
            )
            self.entangle_params = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(attention_dim, attention_dim)) for _ in range(len(qnn_arch) - 1)]
            )
        else:
            self.rotation_params = None
            self.entangle_params = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _self_attention(self, inputs: Tensor, layer: int) -> Tensor:
        """Apply a self‑attention block to a tensor of shape (N, D)."""
        if not self.use_self_attention:
            return inputs
        rot = self.rotation_params[layer]
        ent = self.entangle_params[layer]
        query = torch.matmul(inputs, rot)
        key = torch.matmul(inputs, ent)
        scores = torch.softmax(torch.matmul(query, key.t()) / np.sqrt(self.attention_dim), dim=-1)
        return torch.matmul(scores, inputs)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def forward(self, inputs: Tensor) -> Tensor:
        """Standard forward pass through the hybrid network."""
        x = inputs
        for i, weight in enumerate(self.weights):
            x = torch.tanh(weight @ x.t()).t()
            if self.use_self_attention:
                x = self._self_attention(x, i)
        return x

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts NumPy arrays."""
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        return self.forward(tensor).detach().numpy()

    def __repr__(self) -> str:
        return f"<HybridGraphQNN arch={self.qnn_arch} attention={self.attention_dim} self_attn={self.use_self_attention}>"

__all__ = [
    "HybridGraphQNN",
    "_random_linear",
    "random_training_data",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
