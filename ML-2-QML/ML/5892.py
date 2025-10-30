"""Hybrid classical model combining CNN feature extraction, graph-based message passing,
and a quantum-inspired architecture. This module extends the original QuantumNAT
by adding a graph neural network layer that operates on features derived from
the CNN, using fidelity-based adjacency to guide message passing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
import numpy as np

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared cosine similarity between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float, *,
                       secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def compute_adjacency_batch(features: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return an adjacency matrix (0/1) from pairwise state fidelities across batch."""
    norm = torch.norm(features, dim=1, keepdim=True) + 1e-12
    norm_feat = features / norm
    fidelity_matrix = torch.matmul(norm_feat, norm_feat.t())
    adjacency = (fidelity_matrix >= threshold).float()
    return adjacency

class GraphConvLayer(nn.Module):
    """Simple graph convolution that aggregates neighbor features across batch."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x: (N, in_features), adjacency: (N, N)"""
        agg = torch.matmul(adjacency, x)
        return agg @ self.weight

class HybridNATModel(nn.Module):
    """Classical hybrid model with CNN, graph neural network, and fully connected output."""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 hidden_dim: int = 64,
                 threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, hidden_dim)
        self.graph_conv = GraphConvLayer(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (bsz, 16, 7, 7)
        flattened = features.view(features.size(0), -1)  # (bsz, 16*7*7)
        hidden = F.relu(self.fc1(flattened))  # (bsz, hidden_dim)

        adjacency = compute_adjacency_batch(hidden, self.threshold)  # (bsz, bsz)
        new_hidden = self.graph_conv(hidden, adjacency)  # (bsz, hidden_dim)
        logits = self.fc2(new_hidden)  # (bsz, num_classes)
        return self.norm(logits)

__all__ = ["HybridNATModel"]
