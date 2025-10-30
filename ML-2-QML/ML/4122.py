"""
QuantumGraphFusion: Classical hybrid module combining a fully‑connected layer,
a sampler network, and graph adjacency generation.

This module implements a PyTorch nn.Module that mimics the behaviour of the
original FCL, SamplerQNN and GraphQNN seeds.  The forward pass accepts a
1‑D tensor of parameters (thetas), passes it through a linear layer, then
feeds the result together with a random noise vector into a 2‑layer sampler
network that outputs a probability distribution over a 2‑node graph.  The
resulting adjacency matrix is returned as a numpy array, ready to be fed
into the quantum part.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

__all__ = ["QuantumGraphFusion"]


class QuantumGraphFusion(nn.Module):
    """
    Classical interface of the hybrid module.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_size: int = 4,
        graph_nodes: int = 2,
    ) -> None:
        super().__init__()
        # Linear part (FCL)
        self.linear = nn.Linear(n_features, 1)
        # Sampler network (SamplerQNN)
        self.sampler = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.graph_nodes = graph_nodes

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a probability distribution over a 2‑node
        graph.  The distribution is constructed by feeding the output of the
        linear layer together with a random noise vector into the sampler
        network.
        """
        # Linear layer
        lin_out = torch.tanh(self.linear(thetas))
        # Random noise to diversify sampler input
        noise = torch.rand(2, dtype=lin_out.dtype, device=lin_out.device)
        inp = torch.cat([lin_out, noise])
        probs = F.softmax(self.sampler(inp), dim=-1)
        return probs

    def adjacency_from_probs(self, probs: torch.Tensor) -> np.ndarray:
        """
        Convert a 2‑element probability vector into a 2×2 adjacency matrix.
        The off‑diagonal entry is weighted by the probability of the edge.
        """
        p = probs.detach().cpu().numpy()
        adj = np.eye(self.graph_nodes, dtype=float)
        adj[0, 1] = adj[1, 0] = p[1]  # edge weight
        return adj

    def sample_graph(self, thetas: torch.Tensor) -> nx.Graph:
        """
        Sample a graph adjacency matrix from the forward pass and return a
        NetworkX graph object.
        """
        probs = self.forward(thetas)
        adj = self.adjacency_from_probs(probs)
        g = nx.from_numpy_array(adj)
        return g
