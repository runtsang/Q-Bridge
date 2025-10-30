"""
QuanvolutionHybrid – Classical implementation

This module implements a hybrid classical pipeline that mirrors the
original quanvolution example but augments it with graph‑based
fidelity adjacency.  The design is intentionally modular so that the
class can be swapped with the quantum counterpart in a single codebase.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared cosine similarity between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Iterable[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from feature fidelities."""
    graph = nx.Graph()
    states = list(states)
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class QuanvolutionHybrid(nn.Module):
    """
    Classical quanvolution with optional graph‑based feature aggregation.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 1 for MNIST).
    out_channels : int
        Number of output channels produced by the convolution.
    patch_size : int
        Size of the square patch (default 2).
    stride : int
        Stride of the convolution (default 2).
    graph_threshold : float
        Fidelity threshold for graph edge creation.
    graph_secondary : float | None
        Secondary threshold for weighted edges.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        patch_size: int = 2,
        stride: int = 2,
        graph_threshold: float = 0.8,
        graph_secondary: float | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride)
        # Compute number of patches per image
        self.num_patches = ((28 - patch_size) // stride + 1) ** 2
        self.linear = nn.Linear(out_channels * self.num_patches, 10)
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quanvolution, optional graph aggregation,
        and a linear classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        features = self.conv(x)  # (batch, out_channels, H, W)
        batch_size = features.size(0)
        # Flatten spatial dimensions
        features = features.view(batch_size, -1)  # (batch, out_channels * num_patches)

        # Optional graph aggregation per sample
        if self.graph_threshold > 0.0:
            aggregated = []
            for feat in features:
                # Split into patch features
                patch_feats = feat.view(self.num_patches, -1)  # (num_patches, out_channels)
                graph = fidelity_adjacency(
                    patch_feats,
                    self.graph_threshold,
                    secondary=self.graph_secondary,
                )
                # Compute degree centrality as a simple graph statistic
                degrees = np.array([d for _, d in graph.degree(weight="weight")])
                aggregated.append(torch.from_numpy(degrees).float())
            graph_features = torch.stack(aggregated)  # (batch, num_patches)
            # Concatenate with original features
            features = torch.cat([features, graph_features], dim=1)

        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
