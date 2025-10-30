"""HybridEstimatorQNN: a joint classical‑quantum regressor.

This module fuses concepts from the four reference pairs:
  * Classical feed‑forward regression (EstimatorQNN.py)
  * Fraud‑detection style layers with optional clipping (FraudDetection.py)
  * Self‑attention (SelfAttention.py)
  * Graph‑based fidelity adjacency (GraphQNN.py)

The network consists of fraud‑like layers, a classical self‑attention block,
and a linear output.  Hidden activations are stored so that a graph‑based
regulariser can be applied during training.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import List

# --------------------------------------------------------------------------- #
# Fraud‑like layer
# --------------------------------------------------------------------------- #
class _FraudLayer(nn.Module):
    """Single layer mimicking the photonic fraud‑detection block."""
    def __init__(self, clip: bool = False) -> None:
        super().__init__()
        weight = torch.randn(2, 2, dtype=torch.float32)
        bias = torch.randn(2, dtype=torch.float32)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.randn(2, dtype=torch.float32))
        self.shift = nn.Parameter(torch.randn(2, dtype=torch.float32))
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        if self.clip:
            out = torch.clamp(out, -5.0, 5.0)
        return out

# --------------------------------------------------------------------------- #
# Classical self‑attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention with trainable projections."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
# Graph‑based regulariser
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: List[torch.Tensor], threshold: float) -> nx.Graph:
    """Build a graph from cosine similarities of hidden states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i+1:], start=i+1):
            fid = F.cosine_similarity(a, b, dim=-1).item()
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
    return graph

def laplacian_regulariser(states: List[torch.Tensor], threshold: float) -> torch.Tensor:
    graph = fidelity_adjacency(states, threshold)
    L = nx.laplacian_matrix(graph).astype(float)
    return torch.tensor(L.diagonal().sum(), device=states[0].device)

# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    Joint classical‑quantum regressor.

    Architecture:
      * fraud‑like input layer (no clipping)
      * several fraud‑like hidden layers (with clipping)
      * classical self‑attention
      * linear output
    The forward pass stores hidden activations so that a graph‑based
    regulariser can be applied via ``graph_regulariser``.
    """
    def __init__(
        self,
        hidden_layers: int = 3,
        attention_dim: int = 4,
        adjacency_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_layer = _FraudLayer(clip=False)
        self.hidden_layers = nn.ModuleList(
            [_FraudLayer(clip=True) for _ in range(hidden_layers)]
        )
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.output_layer = nn.Linear(attention_dim, 1)
        self.adjacency_threshold = adjacency_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar regression output."""
        activations = [x]
        h = self.input_layer(x)
        activations.append(h)
        for layer in self.hidden_layers:
            h = layer(h)
            activations.append(h)
        h = self.attention(h)
        activations.append(h)
        out = self.output_layer(h)
        self._activations = activations  # stored for regulariser
        return out

    def graph_regulariser(self) -> torch.Tensor:
        """Compute Laplacian penalty from hidden activations."""
        if not hasattr(self, "_activations"):
            raise RuntimeError("Call forward before computing regulariser.")
        return laplacian_regulariser(self._activations, self.adjacency_threshold)

__all__ = ["HybridEstimatorQNN"]
