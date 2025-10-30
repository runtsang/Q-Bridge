"""Hybrid classical module combining convolution, quantum patch encoding, and graph adjacency.

The module is fully classical and can be imported in a training script.
It exposes two top‑level classes:
  * QuanvolutionGraphHybrid – end‑to‑end model.
  * GraphAdjacencyLayer – optional graph‑based refinement.

The design merges ideas from the original Quanvolution and GraphQNN examples.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class _PatchConv(nn.Module):
    """Classic 2‑D convolution that extracts 2×2 patches and flattens them."""
    def __init__(self, in_channels: int, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class _QuantumPatchEncoder(nn.Module):
    """Classical simulation of a random quantum kernel applied to each 2×2 image patch."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.n_ops = n_ops
        matrix = torch.randn(n_wires, n_wires)
        self.register_buffer('unitary', torch.linalg.qr(matrix)[0])

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, 4)
        encoded = torch.matmul(flat, self.unitary.t())
        return encoded.reshape(B, N, 4).view(B, -1)


class GraphAdjacencyLayer(nn.Module):
    """Builds a weighted graph from feature vector fidelities."""
    def __init__(self, threshold: float = 0.9, secondary: float | None = None, secondary_weight: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    @staticmethod
    def _state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def forward(self, features: Tensor) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(features.size(0)))
        for i, a in enumerate(features):
            for j in range(i + 1, features.size(0)):
                b = features[j]
                fid = self._state_fidelity(a, b)
                if fid >= self.threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif self.secondary is not None and fid >= self.secondary:
                    graph.add_edge(i, j, weight=self.secondary_weight)
        return graph


class QuanvolutionGraphHybrid(nn.Module):
    """End‑to‑end classical network that fuses a patch‑based quantum kernel, a convolutional head, and a graph layer."""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 patch_size: int = 2,
                 conv_out: int = 4,
                 graph_threshold: float = 0.9,
                 graph_secondary: float | None = None):
        super().__init__()
        self.patch_conv = _PatchConv(in_channels, conv_out)
        self.quantum_patch = _QuantumPatchEncoder()
        self.num_patches = (28 // patch_size) ** 2
        self.linear = nn.Linear(conv_out * self.num_patches, num_classes)
        self.graph_layer = GraphAdjacencyLayer(threshold=graph_threshold,
                                               secondary=graph_secondary)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, nx.Graph]:
        patch_features = self.patch_conv(x)
        N_patches = patch_features.size(1) // self.quantum_patch.n_wires
        patch_features = patch_features.view(x.size(0), N_patches, self.quantum_patch.n_wires)
        quantum_features = self.quantum_patch(patch_features)
        logits = self.linear(quantum_features)
        graph = self.graph_layer(quantum_features)
        return F.log_softmax(logits, dim=-1), graph


__all__ = ["QuanvolutionGraphHybrid", "GraphAdjacencyLayer"]
