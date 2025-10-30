"""Hybrid classical graph neural network with quantum‑enhanced refinement.

The module exposes a single factory ``build_graph_classifier`` that
produces a PyTorch ``nn.Module`` mirroring the interface of the quantum
``build_classifier_circuit``.  It constructs a classical GNN backbone
followed by a linear classifier.  The quantum helper returns a
``QuantumCircuit`` that encodes the same node‑embedding dimensionality
and a fidelity‑based adjacency graph for downstream graph‑aware loss
functions.  The two components can be swapped in a training loop
without changing the rest of the pipeline.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=True)


def _node_embedding(
    graph: nx.Graph,
    in_features: int,
    hidden_dim: int,
    num_layers: int,
) -> Tensor:
    """Compute a simple node‑embedding by mean‑pooling neighbors."""
    # gather node features (randomly initialized)
    node_feats = torch.randn((len(graph.nodes), in_features), dtype=torch.float32)
    embeddings = node_feats

    for _ in range(num_layers):
        neigh_sum = torch.zeros_like(embeddings)
        for i, n in enumerate(graph.nodes):
            neigh_sum[i] = embeddings[list(graph.neighbors(n))].sum(dim=0)
        embeddings = F.relu(embeddings + neigh_sum)

    return embeddings


def build_graph_classifier(
    graph: nx.Graph,
    in_features: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical graph‑aware classifier.

    Parameters
    ----------
    graph : nx.Graph
        The graph whose nodes are the input samples.
    in_features : int
        Dimensionality of raw node features.
    hidden_dim : int
        Dimensionality of hidden node embeddings.
    num_layers : int
        Number of message‑passing layers.
    num_classes : int
        Number of output classes (default 2 for binary).
    depth : int
        Depth of the quantum‑style variational circuit used for
        comparison and consistency checks.

    Returns
    -------
    model : nn.Module
        A PyTorch model that accepts a batch of node embeddings
        and outputs logits.
    encoding : Iterable[int]
        The *encoding* indices used by the quantum helper
        (identical to the embedding dimension).
    weight_sizes : Iterable[int]
        The number of trainable parameters in each linear layer.
    observables : List[int]
        The output logits (class indices).
    """

    # 1. Build a simple GNN backbone
    class _GNNBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )

        def forward(self, x: Tensor) -> Tensor:
            out = x
            for layer in self.layers:
                out = F.relu(self.layers[0](out))
            return out

    # 2. Linear classifier head
    class _ClassifierHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(hidden_dim, num_classes)

        def __call__(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # *Construct the full model*
    backbone = _GNNBackbone()
    backbone_input = _node_embedding(graph, in_features, hidden_dim, num_layers)
    backbone_output = backbone(backbone_input)
    # We keep the backbone as a trainable module
    model = nn.Sequential(backbone, _ClassifierHead())

    # *Prepare metadata for quantum comparison*
    encoding = list(range(hidden_dim))
    weight_sizes = [layer.weight.numel() + layer.bias.numel() for layer in model.parameters()]
    observables = list(range(num_classes))

    return model, encoding, weight_sizes, observables


__all__ = ["build_graph_classifier"]
