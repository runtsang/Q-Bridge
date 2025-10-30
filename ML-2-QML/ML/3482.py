"""Classical Graph‑Quanvolution network.

The implementation mirrors the QML interface while staying fully classical.
Key components:
* 2×2 convolutional patch extractor (from QuanvolutionFilter)
* Fidelity‑based adjacency graph built from feature vectors
* Dense graph network with tanh activations
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialized weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared cosine similarity between two vectors."""
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
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    activations: Iterable[Tensor],
) -> List[List[Tensor]]:
    """Propagate activations through a fully‑connected graph network."""
    stored: List[List[Tensor]] = []
    for act in activations:
        layer_vals = [act]
        current = act
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        stored.append(layer_vals)
    return stored

# --------------------------------------------------------------------------- #
# Main module
# --------------------------------------------------------------------------- #

class GraphQuanvolutionNet(nn.Module):
    """Classical Graph‑Quanvolution network.

    Parameters
    ----------
    arch : Sequence[int], optional
        Graph neural network architecture (nodes per layer).  Default
        is ``(4, 8, 16)``.
    threshold : float, optional
        Fidelity threshold for edge creation.  Default ``0.8``.
    """

    def __init__(self, arch: Sequence[int] = (4, 8, 16), threshold: float = 0.8) -> None:
        super().__init__()
        self.arch = arch
        self.threshold = threshold

        # 2×2 convolutional filter (no bias, 4 output channels)
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        # Linear head after graph propagation
        self.linear = nn.Linear(4 * 14 * 14, 10)

        # Random graph‑network weights
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f))
             for in_f, out_f in zip(arch[:-1], arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Extract patches via 2×2 convolution
        features = self.qfilter(x)                      # (B, 4, 14, 14)
        flat = features.view(features.size(0), -1)     # (B, 4*14*14)

        # Build fidelity graph (unused in propagation but kept for analysis)
        _ = fidelity_adjacency(flat, self.threshold)

        # Propagate through the graph network
        activations = feedforward(self.arch, [w for w in self.weights], [flat])

        # Final layer activations → logits
        final = activations[0][-1]
        logits = self.linear(final)
        return F.log_softmax(logits, dim=-1)

__all__ = ["GraphQuanvolutionNet"]
