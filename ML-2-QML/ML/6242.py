"""Hybrid Graph Neural Network for classical training.

This module extends the original GraphQNN with a trainable torch
neural network, a graph‑based regularizer and a simple SGD loop.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn, optim
from torch.nn import functional as F

Tensor = torch.Tensor
State = Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target = weights[-1]
    dataset = random_training_data(target, samples)
    return list(qnn_arch), weights, dataset, target

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        cur = x
        layer_outs: List[Tensor] = [x]
        for w in weights:
            cur = torch.tanh(w @ cur)
            layer_outs.append(cur)
        activations.append(layer_outs)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNClassic(nn.Module):
    """Classic feed‑forward network with optional graph regularizer."""
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    def train_network(self, dataset: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 100, reg_weight: float = 0.0, reg_graph: nx.Graph | None = None):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in dataset:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = F.mse_loss(pred, y)
                if reg_weight > 0.0 and reg_graph is not None:
                    reg = self._graph_regularizer(x, reg_graph)
                    loss += reg_weight * reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(dataset):.4f}")

    def _graph_regularizer(self, x: Tensor, graph: nx.Graph) -> Tensor:
        reg = 0.0
        for i, j in graph.edges():
            xi = x
            xj = x
            reg += 1.0 - state_fidelity(xi, xj)
        return torch.tensor(reg, dtype=x.dtype)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNClassic",
]
