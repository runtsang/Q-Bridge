"""Hybrid Quanvolution network with graph‑based introspection.

This module unifies concepts from the three reference pairs:
- Classical convolution + 2×2 patch extraction (from the original ML seed).
- Quantum kernel applied to each patch with a learnable encoding circuit (from the QML seed).
- A generic classifier factory that can build deep linear heads or a quantum ansatz (from QuantumClassifierModel).
- Fidelity‑based adjacency graphs that can be computed on classical activations or quantum states (from GraphQNN).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "build_classifier_circuit",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuanvolutionHybridNet",
]

# --------------------------------------------------------------------------- #
# 1. Core filter: classical conv + quantum kernel
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extraction followed by flattening."""
    def __init__(self, patch_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=patch_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        patches = self.conv(x)  # (B, 1, H', W')
        return patches.view(x.shape[0], -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier that stacks a QuanvolutionFilter and a linear head."""
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        hidden_sizes: Sequence[int] = (128, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(patch_size, stride)
        layers: List[nn.Module] = []
        in_features = self.filter.conv.out_channels * (28 // stride) * (28 // stride)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.head(feats)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# 2. Classifier factory (classical)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a classical feed‑forward classifier with optional depth."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for classical outputs
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# 3. Forward propagation utilities
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Propagate classical samples through a hand‑crafted linear network."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layer_out = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations

# --------------------------------------------------------------------------- #
# 4. Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity squared between two classical feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from cosine‑based fidelities."""
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
# 5. Convenience wrapper
# --------------------------------------------------------------------------- #

class QuanvolutionHybridNet(nn.Module):
    """Convenience wrapper that exposes the filter, classifier, and utils."""
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        hidden_sizes: Sequence[int] = (128, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(patch_size, stride)
        self.classifier = QuanvolutionClassifier(patch_size, stride, hidden_sizes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
