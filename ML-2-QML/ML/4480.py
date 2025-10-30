"""Combined sampler network with graph utilities and hybrid sigmoid head.

This module merges the classical SamplerQNN, GraphQNN utilities, the hybrid
binary‑classification head, and the Quantum‑NAT style fully connected
projection.  It can be used as a stand‑alone sampler or as a sub‑module
in larger hybrid models.

Key features
------------
* Flexible feed‑forward network with arbitrary hidden layers.
* Differentiable hybrid sigmoid head that can be swapped for a quantum
  expectation layer.
* Graph utilities that build weighted adjacency graphs from state fidelities.
* Convenience routines for generating random networks and synthetic
  training data.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Core utilities – adapted from GraphQNN.py
# --------------------------------------------------------------------------- #

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset where the target is a linear
    transformation of random features."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random MLP and a small training set for its final layer."""
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
    """Return the activations of every layer for each sample."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_vals = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared inner product of two unit‑norm vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid sigmoid head – adapted from ClassicalQuantumBinaryClassification.py
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid with an optional shift that mimics a quantum
    expectation value."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Simple dense head that replaces a quantum circuit in the original model."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --------------------------------------------------------------------------- #
#  SamplerQNNGen161 – the combined class
# --------------------------------------------------------------------------- #

class SamplerQNNGen161(nn.Module):
    """A flexible sampler that combines a classical MLP, graph‑based
    adjacency analysis, and a hybrid sigmoid head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    hidden_dims : Sequence[int]
        Sizes of the hidden layers.
    output_dim : int
        Number of output categories (default 2 for a binary sampler).
    graph_threshold : float
        Fidelity threshold for building the adjacency graph.
    secondary : float | None
        Secondary threshold for weighted edges.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int = 2,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4]
        # Build the feed‑forward part
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        # Hybrid head
        self.hybrid = Hybrid(output_dim)
        # Graph parameters
        self.graph_threshold = graph_threshold
        self.secondary = secondary

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output classes."""
        logits = self.net(inputs)
        probs = self.softmax(logits)
        # Pass the logits through the hybrid head for a binary decision
        binary = self.hybrid(logits)
        # Concatenate the binary probability with its complement
        return torch.cat((binary, 1 - binary), dim=-1)

    # ----------------------------------------------------------------------- #
    #  Utility methods
    # ----------------------------------------------------------------------- #

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample from the categorical distribution produced by the network."""
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples, replacement=True)

    def graph_of_activations(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> nx.Graph:
        """Return a graph where nodes are samples and edges represent
        similarity between the final hidden layer activations."""
        final_states = []
        for sample, _ in samples:
            logits = self.net(sample.unsqueeze(0))
            probs = self.softmax(logits).squeeze(0)
            final_states.append(probs)
        return fidelity_adjacency(
            final_states,
            self.graph_threshold,
            secondary=self.secondary,
        )

    @staticmethod
    def generate_random(
        arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Convenience wrapper around the GraphQNN.random_network function."""
        return random_network(arch, samples)

    @staticmethod
    def generate_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Convenience wrapper around the GraphQNN.random_training_data function."""
        return random_training_data(weight, samples)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Return the squared inner product of two probability vectors."""
        return state_fidelity(a, b)

    __all__ = [
        "SamplerQNNGen161",
        "HybridFunction",
        "Hybrid",
        "random_network",
        "random_training_data",
        "feedforward",
        "state_fidelity",
        "fidelity_adjacency",
    ]
