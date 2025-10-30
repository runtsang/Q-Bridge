"""Hybrid classical graph neural network that integrates quantum‑inspired
fidelity and self‑attention.  The implementation builds on the
original GraphQNN, QuantumRegression and SelfAttention seeds, but
extends them with a graph‑aware attention weighting scheme."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Import helper utilities from the seed modules
from GraphQNN import _random_linear as _rand_linear
from GraphQNN import random_training_data as _rand_train
from GraphQNN import fidelity_adjacency as _fid_adj
from QuantumRegression import generate_superposition_data as _gen_data
from SelfAttention import SelfAttention as _SA_cls


class GraphQNNHybrid(nn.Module):
    """Hybrid classical graph neural network.

    Combines:
    * Random linear layers (from GraphQNN) to propagate features.
    * Fidelity‑based adjacency (from GraphQNN) to capture state overlap.
    * Self‑attention weighting (from SelfAttention) to modulate edge
      importance.
    """
    def __init__(self, qnn_arch: Sequence[int], attention_dim: int = 4):
        super().__init__()
        self.arch = list(qnn_arch)
        self.weights = [_rand_linear(in_f, out_f)
                        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        self.attn = _SA_cls()  # returns a ClassicalSelfAttention instance

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layers."""
        activations = [features]
        x = features
        for w in self.weights:
            x = torch.tanh(w @ x)
            activations.append(x)
        return activations

    def build_graph(self, activations: List[torch.Tensor],
                    threshold: float = 0.8) -> nx.Graph:
        """Merge fidelity and attention into a weighted graph."""
        # Convert activations to numpy for fidelity calculation
        states = [act.detach().cpu().numpy() for act in activations]
        # Fidelity graph
        fid_graph = _fid_adj(states, threshold)

        # Attention matrix (symmetric, shape N x N)
        # Use random parameters to illustrate the interface
        rot_params = np.random.randn(3 * self.attn.embed_dim * self.attn.embed_dim)
        ent_params = np.random.randn(self.attn.embed_dim - 1)
        attn_scores = self.attn.run(rot_params, ent_params, states[0])

        # Build attention graph
        attn_graph = nx.Graph()
        attn_graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                attn_graph.add_edge(i, j, weight=attn_scores[i, j])

        # Merge graphs with a convex combination
        merged = nx.Graph()
        merged.add_nodes_from(fid_graph.nodes(data=True))
        for u, v, w in fid_graph.edges(data=True):
            merged.add_edge(u, v, weight=w["weight"])
        for u, v, w in attn_graph.edges(data=True):
            if merged.has_edge(u, v):
                merged[u][v]["weight"] = 0.7 * merged[u][v]["weight"] + 0.3 * w["weight"]
            else:
                merged.add_edge(u, v, weight=w["weight"] * 0.3)
        return merged

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(preds, targets)


class RegressionDataset(Dataset):
    """Dataset that mirrors QuantumRegression.py but returns torch tensors."""
    def __init__(self, samples: int, num_features: int):
        self.x, self.y = _gen_data(num_features, samples)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.x[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate random weights and training data for the hybrid model."""
    weights = [_rand_linear(in_f, out_f)
               for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    train_data = _rand_train(target_weight, samples)
    return qnn_arch, weights, train_data, target_weight


__all__ = [
    "GraphQNNHybrid",
    "RegressionDataset",
    "generate_superposition_data",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
]
