"""GraphQNN__gen275: Classical GCN with optional quantum layer.

The module retains the original feed‑forward, fidelity and adjacency helpers
while adding a lightweight GCN backbone.  The feedforward routine now
performs graph convolution, producing per‑layer node embeddings and a
global latent vector.  The rest of the interface (random_network,
random_training_data, state_fidelity, fidelity_adjacency) remains
compatible with the seed implementation.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical GCN implementation
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    """Single graph convolutional layer using mean aggregation."""

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_feats, in_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        # adjacency: [N, N]
        agg = torch.matmul(adjacency, features)  # [N, in_feats]
        out = torch.matmul(agg, self.weight.t())  # [N, out_feats]
        out = out + self.bias
        return F.relu(out)

class GraphQNN__gen275(nn.Module):
    """Hybrid graph neural network with a classical GCN backbone.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths: first element is input feature dimension,
        subsequent elements are output dimensions for each GCN layer.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.layers.append(GCNLayer(in_f, out_f))

    def forward(self, adjacency: Tensor, features: Tensor) -> List[Tensor]:
        """Return per‑layer node embeddings and a global embedding.

        Returns list with length len(arch)+1:
          - element 0: input node features
          - element i (i>=1): node embeddings after layer i
          - final element: mean of last node embeddings (global vector)
        """
        activations: List[Tensor] = [features]
        current = features
        for layer in self.layers:
            current = layer(adjacency, current)
            activations.append(current)
        # Global embedding as mean pooling
        global_emb = current.mean(dim=0, keepdim=True)  # shape [1, out_dim]
        activations.append(global_emb.squeeze(0))  # shape [out_dim]
        return activations

    def parameters_vector(self) -> List[Tensor]:
        """Return all learnable parameters as a flat list."""
        return [p for p in self.parameters()]

# --------------------------------------------------------------------------- #
#  Utility functions (backwards compatible)
# --------------------------------------------------------------------------- #
def random_training_data(
    model: GraphQNN__gen275,
    samples: int,
    num_nodes: int = 5,
    seed: int | None = None,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs.

    For each sample:
      * Random node features (num_nodes × input_dim)
      * Target global embedding computed by the *random* model.
    """
    rng = np.random.default_rng(seed)
    data: List[Tuple[Tensor, Tensor]] = []
    in_dim = model.arch[0]
    for _ in range(samples):
        features = torch.tensor(
            rng.normal(size=(num_nodes, in_dim)), dtype=torch.float32
        )
        # Random adjacency (symmetric, no self‑loops)
        adj_mat = rng.integers(0, 2, size=(num_nodes, num_nodes))
        adjacency = torch.tensor(adj_mat, dtype=torch.float32)
        adjacency = (adjacency + adjacency.t()) / 2
        torch.fill_diagonal_(adjacency, 0)
        deg = adjacency.sum(dim=1, keepdim=True) + 1e-8
        adjacency = adjacency / deg
        # Compute target global embedding
        _, _, _, global_emb = _forward_and_target(model, adjacency, features)
        data.append((features, global_emb))
    return data

def _forward_and_target(
    model: GraphQNN__gen275,
    adjacency: Tensor,
    features: Tensor,
) -> Tuple[List[Tensor], None, None, Tensor]:
    """Compute activations and target global embedding for a single sample."""
    activations = model(adjacency, features)
    global_emb = activations[-1]
    return activations, None, None, global_emb

def random_network(
    arch: Sequence[int],
    samples: int,
    num_nodes: int = 5,
    seed: int | None = None,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random GCN model, training set and target parameters.

    Returns
      * arch
      * list of weight matrices (flattened)
      * training data
      * target weight of last layer
    """
    rng = np.random.default_rng(seed)
    model = GraphQNN__gen275(arch)
    # Randomise model weights
    for layer in model.layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    target_weight = model.layers[-1].weight.data.clone()
    training_data = random_training_data(model, samples, num_nodes, seed)
    # Extract weights as a list of raw tensors
    weights = [layer.weight.data.clone() for layer in model.layers]
    return list(arch), weights, training_data, target_weight

def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Perform feed‑forward using provided weights.

    Parameters
    ----------
    arch: Sequence[int]
        Architecture of the GCN.
    weights: Sequence[Tensor]
        List of weight matrices for each layer.
    samples: Iterable[Tuple[Tensor, Tensor]]
        Each sample is a pair (features, target).  Only the feature part is used.

    Returns
    -------
    List[List[Tensor]]
        For each sample a list of node embeddings per layer and the global
        embedding.
    """
    # Build a temporary model with frozen weights
    model = GraphQNN__gen275(arch)
    for layer, w in zip(model.layers, weights):
        layer.weight.data = w.clone()
        nn.init.zeros_(layer.bias)
    activations_per_sample: List[List[Tensor]] = []

    for features, _ in samples:
        num_nodes = features.shape[0]
        rng = np.random.default_rng()
        adj_mat = rng.integers(0, 2, size=(num_nodes, num_nodes))
        adjacency = torch.tensor(adj_mat, dtype=torch.float32)
        adjacency = (adjacency + adjacency.t()) / 2
        torch.fill_diagonal_(adjacency, 0)
        deg = adjacency.sum(dim=1, keepdim=True) + 1e-8
        adjacency = adjacency / deg
        layerwise = model(adjacency, features)
        activations_per_sample.append(layerwise)
    return activations_per_sample

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two vectorised states."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "GraphQNN__gen275",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
