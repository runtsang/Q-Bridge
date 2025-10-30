"""Hybrid classical estimator combining graph‑style layers and self‑attention.

The module defines a single class ``SharedClassName`` that can be used
as a drop‑in replacement for the original EstimatorQNN.  It inherits
from :class:`torch.nn.Module` and optionally inserts a self‑attention
block after each hidden layer.  The architecture is specified by
``qnn_arch`` – a sequence of integers describing the width of each
layer.  The implementation also provides helper functions that
mirror the utilities from the original GraphQNN module, enabling
easy generation of synthetic data and fidelity‑based graph analysis.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Iterable, Tuple, List
import numpy as np

# --------------------------------------------------------------------------- #
#  Self‑attention sub‑module
# --------------------------------------------------------------------------- #
class SelfAttentionModule(nn.Module):
    """
    A lightweight self‑attention block operating on a batch of vectors.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input vectors.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear projections for query, key and value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape.
        """
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
#  Hybrid feed‑forward network
# --------------------------------------------------------------------------- #
class SharedClassName(nn.Module):
    """
    Classical estimator that mimics the structure of the quantum
    EstimatorQNN while adding a graph‑style feed‑forward backbone
    and optional self‑attention.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer widths, e.g. ``[2, 8, 4, 1]``.
    use_attention : bool, default=True
        Whether to insert a SelfAttentionModule after each hidden
        layer.
    """

    def __init__(self, qnn_arch: Sequence[int], use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.layers: List[nn.Module] = []

        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            linear = nn.Linear(in_f, out_f, bias=False)
            self.layers.append(linear)
            self.layers.append(nn.Tanh())
            if use_attention and out_f > 1:
                self.layers.append(SelfAttentionModule(out_f))
        # Final linear to a single output
        self.output = nn.Linear(qnn_arch[-1], 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch, 1)``.
        """
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# --------------------------------------------------------------------------- #
#  Utility functions – mirror GraphQNN helpers
# --------------------------------------------------------------------------- #
def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic training data given a target weight matrix.

    Parameters
    ----------
    weight : torch.Tensor
        Target linear transformation ``(out, in)``.
    samples : int
        Number of data points to generate.

    Returns
    -------
    List[Tuple[torch.Tensor, torch.Tensor]]
        List of (features, target) pairs.
    """
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target   = weight @ features
        dataset.append((features, target))
    return dataset

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Run a feed‑forward pass through the classical network and
    capture intermediate activations.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the network.
    weights : Sequence[torch.Tensor]
        List of weight matrices for each linear layer.
    samples : Iterable[Tuple[torch.Tensor, torch.Tensor]]
        Iterable of (features, target) pairs.

    Returns
    -------
    List[List[torch.Tensor]]
        Activations per sample: outer list over samples,
        inner list over layers.
    """
    activations = []
    for features, _ in samples:
        layerwise = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute the squared overlap between two vectors.

    Parameters
    ----------
    a, b : torch.Tensor
        Vectors of equal dimensionality.

    Returns
    -------
    float
        Fidelity value in ``[0, 1]``.
    """
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from state fidelities.

    Parameters
    ----------
    states : Sequence[torch.Tensor]
        List of state vectors.
    threshold : float
        Primary fidelity threshold.
    secondary : float, optional
        Secondary threshold for weaker edges.
    secondary_weight : float, default=0.5
        Weight assigned to secondary edges.

    Returns
    -------
    networkx.Graph
        Graph with nodes as state indices and weighted edges.
    """
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = ["SharedClassName", "random_training_data", "feedforward",
           "state_fidelity", "fidelity_adjacency"]
