"""Hybrid graph‑quantum neural network with a classical analogue.

This module defines a shared ``GraphQuanvolutionHybrid`` class that implements
two complementary sub‑components:
* A **classical graph neural network** that operates on node features and
  weighted edges derived from state fidelities.
* A **classical convolution‑style filter** that mimics the quantum
  two‑qubit kernel used in the quanvolution example and produces a
  feature vector which is then fed into a linear classifier.

The design is inspired by the two reference seeds – the GraphQNN
implementation (graph‑based state propagation) and the Quanvolution
example (two‑qubit quantum kernel).  The class is fully
implementable with PyTorch, NumPy and NetworkX, and it exposes a
``forward`` method that accepts either a batch of images (``torch.Tensor``)
or a graph structure, allowing the user to *interpolate* between the
convolutional and graph‑based inference.
"""

from __future__ import annotations

import itertools
import math
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical Graph Neural Network utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic training data for a linear target."""
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    """Construct a toy multi‑layer GNN with random weights and a training set."""
    weights: list[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight

def feedforward(
    qnn_arch: list[int],
    weights: list[torch.Tensor],
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[list[torch.Tensor]]:
    """Run a forward pass through the toy GNN and store activations."""
    activations: list[list[torch.Tensor]] = []
    for features, _ in samples:
        current = features
        layerwise = [current]
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return squared overlap of two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: list[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# 2. Classical “quanvolution” filter
# --------------------------------------------------------------------------- #
class ClassicalQuanvolutionFilter(nn.Module):
    """A deterministic 2×2 patch extractor followed by a linear projection."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the output of the convolution into a 1‑D feature vector."""
        return self.conv(x).view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# 3. Hybrid model
# --------------------------------------------------------------------------- #
class GraphQuanvolutionHybrid(nn.Module):
    """Hybrid model that can operate on images or graph data.

    Parameters
    ----------
    qnn_arch : list[int]
        Architecture for the toy GNN (node feature dimension per layer).
    patch_size : int, optional
        Size of the square patch for the quanvolution filter.
    """

    def __init__(self, qnn_arch: list[int], patch_size: int = 2):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.gnn_weights, _, _, _ = random_network(qnn_arch, samples=1)
        self.quanvolution = ClassicalQuanvolutionFilter(patch_size=patch_size)
        # Classifier head that matches the flattened output of the quanvolution
        output_dim = (28 // patch_size) ** 2 * self.quanvolution.conv.out_channels
        self.classifier = nn.Linear(output_dim, 10)

    def forward(self, x: torch.Tensor | nx.Graph) -> torch.Tensor:
        """Dispatch based on the input type."""
        if isinstance(x, nx.Graph):
            return self._forward_graph(x)
        elif isinstance(x, torch.Tensor):
            return self._forward_image(x)
        else:
            raise TypeError("Input must be a torch.Tensor or networkx.Graph")

    def _forward_image(self, img: torch.Tensor) -> torch.Tensor:
        """Apply quanvolution followed by a linear head."""
        features = self.quanvolution(img)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def _forward_graph(self, G: nx.Graph) -> torch.Tensor:
        """Compute node embeddings using the toy GNN and aggregate."""
        # Initialise node features as the degree vector
        node_feats = torch.tensor([G.degree(n) for n in G.nodes()], dtype=torch.float32)
        # Run the toy GNN
        activations = feedforward(self.qnn_arch, self.gnn_weights, [(node_feats, node_feats)])
        # Aggregate last layer into a graph representation
        graph_emb = activations[-1][0]
        # Classify the graph embedding
        logits = self.classifier(graph_emb.unsqueeze(0))
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "GraphQuanvolutionHybrid",
    "ClassicalQuanvolutionFilter",
    "fidelity_adjacency",
    "random_network",
    "feedforward",
]
