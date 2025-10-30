"""Graph-based quantum neural network with attention, convolution, and autoencoder.
This module merges GraphQNN, SelfAttention, QCNN, and Autoencoder concepts into a
single PyTorch‑based architecture that can be trained classically or used as a
reference for its quantum counterpart.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor

# --- Utility functions from original GraphQNN --------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --- Classical self‑attention (adapted from SelfAttention.py) ----------------

class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        query = inputs @ self.rotation
        key   = inputs @ self.entangle
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

# --- QCNN‑style convolution blocks -----------------------------------------

class QCNNBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

# --- Autoencoder component -----------------------------------------------

class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

# --- Main integrated architecture -----------------------------------------

class GraphQNNGen111(nn.Module):
    """
    Combined graph‑based quantum neural network that incorporates:
      * linear layers (GraphQNN)
      * self‑attention (SelfAttention)
      * convolution‑like blocks (QCNN)
      * autoencoder latent representation
    """
    def __init__(self,
                 qnn_arch: Sequence[int],
                 attention_dim: int = 4,
                 conv_layers: int = 3,
                 autoencoder_latent: int = 32,
                 autoencoder_hidden: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.arch = list(qnn_arch)

        # GraphQNN linear layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ])

        # Self‑attention
        self.attention = ClassicalSelfAttention(attention_dim)

        # QCNN blocks
        self.conv_blocks = nn.ModuleList([
            QCNNBlock(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ][:conv_layers])

        # Autoencoder
        self.autoencoder = AutoencoderNet(
            input_dim=qnn_arch[-1],
            latent_dim=autoencoder_latent,
            hidden_dims=autoencoder_hidden,
            dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        # attention
        x = self.attention(x)

        # linear + conv
        for lin, conv in zip(self.linear_layers, self.conv_blocks):
            x = conv(lin(x))

        # autoencoder latent
        latent = self.autoencoder.encode(x)
        return latent

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations per layer for a batch of samples."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            # attention
            current = self.attention(current)
            activations.append(current)
            # linear + conv
            for lin, conv in zip(self.linear_layers, self.conv_blocks):
                current = conv(lin(current))
                activations.append(current)
            # latent
            latent = self.autoencoder.encode(current)
            activations.append(latent)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen111.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "GraphQNNGen111",
    "random_network",
    "random_training_data",
]
