"""HybridBinaryClassifier: classical baseline for the quantum hybrid model.

The class integrates a convolutional backbone, a graph‑based relational layer,
and a photonic‑inspired transform before a sigmoid output.  It mirrors the
structure of the original QCNet while adding graph reasoning and a
classical analog of the photonic fraud‑detection layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List

# --- Utility functions ---------------------------------------------------------

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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

# --- Graph convolution layer ----------------------------------------------------

class GraphConvolution(nn.Module):
    """Simple message‑passing: weighted sum of neighbors."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features

    def forward(self, features: torch.Tensor, adjacency: nx.Graph) -> torch.Tensor:
        out = torch.zeros_like(features)
        for node in adjacency.nodes():
            neighbors = list(adjacency.neighbors(node))
            if neighbors:
                weights = torch.tensor(
                    [adjacency[node][nbr]["weight"] for nbr in neighbors],
                    dtype=features.dtype,
                    device=features.device,
                )
                neighbor_feats = features[neighbors]
                out[node] = torch.sum(weights.unsqueeze(1) * neighbor_feats, dim=0)
            else:
                out[node] = features[node]
        return out

# --- Photonic‑inspired layer ----------------------------------------------------

class PhotonicLayer(nn.Module):
    """Linear → tanh → scaling + shift (mimics the photonic fraud‑detection layer)."""

    def __init__(self, in_features: int, out_features: int,
                 scale: torch.Tensor | None = None,
                 shift: torch.Tensor | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(scale if scale is not None else torch.ones(out_features))
        self.shift = nn.Parameter(shift if shift is not None else torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear(x))
        x = x * self.scale + self.shift
        return x

# --- Hybrid binary classifier ----------------------------------------------------

class HybridBinaryClassifier(nn.Module):
    """CNN + graph reasoning + photonic layer + sigmoid output."""

    def __init__(self,
                 in_channels: int = 3,
                 graph_threshold: float = 0.8,
                 photonic_scale: torch.Tensor | None = None,
                 photonic_shift: torch.Tensor | None = None) -> None:
        super().__init__()
        self.graph_threshold = graph_threshold

        # CNN backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Graph convolution over batch samples
        self.graph_conv = GraphConvolution(1)

        # Photonic‑inspired transform
        self.photonic = PhotonicLayer(1, 1, photonic_scale, photonic_shift)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # (batch,)

        # Build adjacency across the batch
        adjacency = fidelity_adjacency(
            [x[i].unsqueeze(0) for i in range(x.size(0))],
            self.graph_threshold,
        )

        # Graph‑based message passing
        x = self.graph_conv(x.unsqueeze(1), adjacency).squeeze(1)

        # Photonic‑inspired layer
        x = self.photonic(x.unsqueeze(1)).squeeze(1)

        # Sigmoid output
        return self.sigmoid(x)

__all__ = ["HybridBinaryClassifier"]
