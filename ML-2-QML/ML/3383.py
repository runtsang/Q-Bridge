"""Hybrid classical graph neural network with quanvolution preprocessing."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Cosine‑like fidelity for real tensors."""
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
    """Build weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extraction followed by 2‑D convolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x shape: (batch, channels, H, W)
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using the quanvolution filter and a linear head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 28×28 image → 14×14 patches → 4 channels each
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class GraphQNNHybrid(nn.Module):
    """Classical graph neural network that first applies quanvolution preprocessing."""
    def __init__(self, qnn_arch: Sequence[int], in_features: int = 4) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.layers: nn.ModuleList = nn.ModuleList()
        # Build linear layers according to architecture
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.qfilter = QuanvolutionFilter(in_channels=in_features, out_channels=4)

    def forward(self, x: Tensor, adjacency: nx.Graph) -> Tensor:
        """
        x: node feature matrix (batch, nodes, features)
        adjacency: graph adjacency (used only for potential GNN message passing; omitted here for brevity)
        """
        # Apply quanvolution to each node feature vector (treated as 2×2 patch)
        batch, nodes, _ = x.shape
        # Reshape to simulate 2×2 image patches per node
        patches = x.view(batch * nodes, 1, 2, 2)
        qfeat = self.qfilter(patches)  # (batch*nodes, 4)
        qfeat = qfeat.view(batch, nodes, -1)

        out = qfeat
        for layer in self.layers:
            out = F.relu(layer(out))
        return out

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate random weights and synthetic training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target_weight = weights[-1]
        # Synthetic data: features → target via target_weight
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1))
            target = target_weight @ features
            dataset.append((features, target))
        return list(qnn_arch), weights, dataset, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic training data for a given weight matrix."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

__all__ = [
    "state_fidelity",
    "fidelity_adjacency",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "GraphQNNHybrid",
]
