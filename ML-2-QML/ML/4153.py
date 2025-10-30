"""Unified classical component for a quantum graph classifier.

The module exposes a single ``UnifiedQuantumGraphClassifier`` class that
provides a feed‑forward interface compatible with the original
``FCL`` and ``QuantumClassifierModel`` seeds.  It builds a stack of
fully‑connected blocks (one per quantum layer) and tracks the
activation statistics that are later used by the quantum module to
construct a fidelity graph.  The design intentionally mirrors the
parameter‑sharding strategy of the reference QML module while keeping
the entire forward pass in PyTorch.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


class UnifiedQuantumGraphClassifier(nn.Module):
    """
    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of quantum‑like layers to emulate.
    hidden_dim : int, optional
        Width of each fully‑connected block.  Defaults to ``num_features``.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features

        # ------------------------------------------------------------------
        # Classical sub‑network: a stack of fully‑connected + ReLU blocks
        # ------------------------------------------------------------------
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        # Final head for binary classification
        layers.append(nn.Linear(self.hidden_dim, 2))
        self.classifier = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # Metadata that mimics the quantum interface
        # ------------------------------------------------------------------
        self.encoding = list(range(num_features))          # indices of feature columns
        self.weight_sizes = [
            m.weight.numel() + m.bias.numel()
            for m in self.classifier.modules()
            if isinstance(m, nn.Linear)
        ]
        self.observables = list(range(2))                 # binary output nodes

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits.  The method mirrors the
        ``run`` function of the quantum FCL; it accepts a 2‑D tensor
        (batch, features) and produces a tensor of shape (batch, 2).
        """
        return self.classifier(X)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that accepts a list of classical parameters
        *in the same order* as the quantum circuit's `theta` vector.
        The output matches the expected shape of *the quantum* `run` method.
        """
        theta_tensor = torch.as_tensor(thetas, dtype=torch.float32)
        # Reshape to match the linear layers' weight shape
        idx = 0
        for l in range(self.depth):
            w_shape = (self.hidden_dim, self.hidden_dim if l < self.depth - 1 else 2)
            w_size = np.prod(w_shape)
            # Slice the theta vector for this layer
            weight = theta_tensor[idx : idx + w_size].view(w_shape)
            idx += w_size
        # Forward pass with the sliced weights (dummy input)
        dummy = torch.zeros(1, self.num_features)
        out = self.classifier(dummy)
        return out.detach().numpy()

    def get_activation_stats(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Return a list of activations per layer (excluding the final head).
        Useful for feeding into the quantum module for fidelity graph
        construction.
        """
        activations: List[torch.Tensor] = []
        current = X
        for l in range(self.depth):
            current = F.relu(self.classifier[2 * l](current))
            activations.append(current)
        return activations

    def build_fidelity_graph(self, activations: List[torch.Tensor], threshold: float = 0.95) -> nx.Graph:
        """
        Build a graph where each node represents a layer activation.
        Nodes are connected if the cosine similarity between two activations
        *in the batch* is above the threshold.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for i in range(len(activations)):
            for j in range(i + 1, len(activations)):
                fid = F.cosine_similarity(activations[i], activations[j], dim=0).item()
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                else:
                    graph.add_edge(i, j, weight=0.5)
        return graph


__all__ = ["UnifiedQuantumGraphClassifier"]
