"""Hybrid quanvolution classifier with graph‑regularised classical filter.

This module replaces the original Quanvolution.py.  It
incorporates graph‑based smoothing inspired by GraphQNN and
parameter clipping concepts from FraudDetection.
"""

from __future__ import annotations

import itertools
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# --- Utility functions (from GraphQNN) ---------------------------------------

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared inner product of two normalised feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Iterable[torch.Tensor],
    threshold: float = 0.8,
    *,
    secondary: float | None = 0.5,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    states = list(states)
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Core model -------------------------------------------------------------

class HybridQuanvolutionFilter(nn.Module):
    """Classical 2×2 quanvolution filter with Laplacian smoothing."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        graph_threshold: float = 0.8,
        secondary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)  # (B, C, H', W')
        B, C, H, W = feats.shape
        patches = feats.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        smoothed = []
        for b in range(B):
            states = patches[b]  # (N, C)
            graph = fidelity_adjacency(
                states,
                self.graph_threshold,
                secondary=self.secondary_threshold,
            )
            L = nx.laplacian_matrix(graph).toarray()
            smooth = torch.from_numpy((torch.eye(L.shape[0]) + torch.from_numpy(L)).float())
            smoothed.append(torch.matmul(states, smooth))
        smoothed = torch.stack(smoothed)  # (B, N, C)
        return smoothed.view(B, -1)  # flatten

class HybridQuanvolutionClassifier(nn.Module):
    """Classifier using the hybrid filter followed by a linear head."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        graph_threshold: float = 0.8,
        secondary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter(
            in_channels,
            out_channels=4,
            kernel_size=2,
            stride=2,
            graph_threshold=graph_threshold,
            secondary_threshold=secondary_threshold,
        )
        # MNIST 28×28 → 14×14 patches × 4 channels
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionFilter", "HybridQuanvolutionClassifier"]
