"""Hybrid classical graph‑neural‑network + quantum‑inspired architecture.

This module builds on the original GraphQNN utilities and the
Quantum‑NAT CNN+FC model.  The `GraphQNNGen121` class first
encodes node features with a shallow CNN, then performs a
graph‑convolution using a fidelity‑based adjacency matrix.
Finally a linear projection produces a 4‑dimensional output that
can be compared against a quantum counterpart.
"""

from __future__ import annotations

import networkx as nx
from typing import Sequence

import torch
import torch.nn as nn

from.GraphQNN import (
    fidelity_adjacency,
)

class GraphQNNGen121(nn.Module):
    """Classical hybrid GNN + CNN model.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the underlying graph neural network.
    cnn_channels : int, optional
        Number of channels in the first convolutional layer.
    """

    def __init__(self, qnn_arch: Sequence[int], cnn_channels: int = 8) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.graph_proj = nn.Linear(16 * 7 * 7, qnn_arch[0])
        self.graph_layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.graph_layers.append(nn.Linear(in_f, out_f))
        self.final = nn.Linear(qnn_arch[-1], 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input images of shape (B, 1, 28, 28).
        """
        # CNN feature extraction
        features = self.cnn(x)
        flat = features.view(x.size(0), -1)
        node_vec = self.graph_proj(flat)

        # Fidelity‑based adjacency over the batch
        adj = fidelity_adjacency(
            [node_vec[i] for i in range(node_vec.size(0))],
            threshold=0.9,
        )
        mat = torch.tensor(
            nx.to_numpy_array(adj),
            dtype=torch.float32,
            device=node_vec.device,
        )

        # Graph convolution: linear mix followed by message passing
        out = node_vec
        for layer in self.graph_layers:
            out = layer(out)
            out = torch.matmul(mat, out)

        out = self.final(out)
        return self.norm(out)

__all__ = ["GraphQNNGen121"]
