"""Hybrid Quanvolution AutoEncoder with Graph-based Quantum Features.

This module defines a class `HybridQuanvolutionAutoEncoderGraph` that
combines a classical quanvolution filter, a variational quantum
auto‑encoder (implemented in the `qml` module), and a graph neural
network that uses fidelity-based adjacency derived from the quantum
latent states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from.qml import VariationalQuantumAutoEncoder, QuantumGraphNeuralNetwork

class ClassicalQuanvolutionFilter(nn.Module):
    """Convolutional filter that extracts 2×2 patches from a 28×28 image."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        features = self.conv(x)  # (B, 4, 14, 14)
        return features.view(x.shape[0], -1)  # (B, 4*14*14)

class HybridQuanvolutionAutoEncoderGraph(nn.Module):
    """Hybrid classical‑quantum architecture."""
    def __init__(self,
                 latent_dim: int = 3,
                 num_trash: int = 2,
                 graph_threshold: float = 0.8,
                 graph_secondary: float | None = None,
                 graph_secondary_weight: float = 0.5) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.q_autoencoder = VariationalQuantumAutoEncoder(
            latent_dim=latent_dim,
            num_trash=num_trash
        )
        self.qnn = QuantumGraphNeuralNetwork(
            graph_threshold=graph_threshold,
            graph_secondary=graph_secondary,
            graph_secondary_weight=graph_secondary_weight
        )
        latent_size = 2 ** latent_dim
        self.classifier = nn.Linear(784 + latent_size * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.filter(x)  # (B, 784)
        # Quantum autoencoder: returns latent statevectors (B, 2**latent_dim, 2)
        latent = self.q_autoencoder(features)  # complex amplitudes
        # Flatten latent amplitudes into a real vector
        latent_flat = latent.view(latent.size(0), -1)  # (B, 2**latent_dim * 2)
        # Build graph from latent states
        graph = self.qnn(latent)  # returns nx.Graph
        # Use graph adjacency to produce node embeddings
        adjacency = torch.tensor(nx.to_numpy_array(graph), dtype=features.dtype, device=features.device)
        node_embeds = adjacency @ latent_flat
        # Concatenate latent and graph embeddings
        combined = torch.cat([latent_flat, node_embeds], dim=1)
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAutoEncoderGraph", "ClassicalQuanvolutionFilter"]
