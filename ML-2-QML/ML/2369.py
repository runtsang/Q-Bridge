"""QuantumHybridGraphNet – classical implementation.

This module implements a hybrid neural network that combines:
  * a 2‑D CNN backbone identical to the original seed,
  * a classical dense head that produces a single logit,
  * a fidelity‑based graph generator that builds a weighted adjacency
    graph from the logit values of a batch.
The class is fully PyTorch‑based and uses networkx for graph
operations.  No quantum libraries are required.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical CNN backbone – identical to the original seed
# --------------------------------------------------------------------------- #
class _CNNBackbone(nn.Module):
    """2‑D convolutional feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # The linear layer size matches the flattened feature dimension
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.fc3(x)  # shape: (batch, 1)

# --------------------------------------------------------------------------- #
#  Fidelity‑based graph generator
# --------------------------------------------------------------------------- #
class _FidelityGraphGenerator:
    """Builds a graph from pairwise fidelities of scalar logits.

    The method accepts a batch of logits (one per sample)
    and returns a weighted adjacency matrix (nx.Graph).
    """

    def __init__(self, threshold: float = 0.8, secondary: float = 0.6):
        self.threshold = threshold
        self.secondary = secondary

    def __call__(self, logits: torch.Tensor) -> nx.Graph:
        """Return a graph where nodes are samples and edges are weighted
        by the similarity of their logits.
        """
        logits_np = logits.detach().cpu().numpy().flatten()
        graph = nx.Graph()
        graph.add_nodes_from(range(len(logits_np)))
        for i, j in itertools.combinations(range(len(logits_np)), 2):
            # Simple similarity: 1 - |diff| / max_diff
            diff = abs(logits_np[i] - logits_np[j])
            max_diff = max(logits_np) - min(logits_np) + 1e-12
            similarity = 1.0 - diff / max_diff
            if similarity >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and similarity >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary)
        return graph

# --------------------------------------------------------------------------- #
#  Main hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridGraphNet(nn.Module):
    """Classical hybrid model that fuses a CNN backbone, a linear head
    and a fidelity‑based graph generator.

    The class returns both the class logits and the adjacency graph
    constructed from the logits.  The graph can be used for
    message‑passing or contrastive training downstream.
    """

    def __init__(self, threshold: float = 0.8, secondary: float = 0.6) -> None:
        super().__init__()
        self.backbone = _CNNBackbone()
        self.classifier = nn.Linear(1, 2)
        self.graph_generator = _FidelityGraphGenerator(threshold, secondary)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nx.Graph]:
        """Return logits and adjacency graph."""
        features = self.backbone(x)
        logits = self.classifier(features)
        graph = self.graph_generator(logits)
        return logits, graph

__all__ = ["QuantumHybridGraphNet"]
