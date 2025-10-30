"""Hybrid classical GNN with trainable weights and graph fidelity.

The original seed implements a random‑weight network.  This new
module turns the network into a learnable PyTorch module, adds a
gradient‑based training helper, and exposes a graph‑based fidelity
metric for the final layer.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "GraphQNN",
    "random_graph",
    "random_graph_with_params",
    "train_step",
    "build_graph",
]

class GraphQNN(nn.Module):
    """Parameter‑tied graph neural network.

    The network is defined by an integer sequence ``qnn_arch`` where
    each entry is the number of nodes in that layer.  All nodes in a
    layer share the same weight matrix.  The network is fully
    differentiable and can be trained with any PyTorch optimiser.
    """

    def __init__(self, *qnn_arch: int, device: str = "cpu"):
        super().__init__()
        if len(qnn_arch) < 2:
            raise ValueError("Architecture must contain at least two layers")
        self.arch = list(qnn_arch)
        self.device = device
        self.weights: List[nn.Parameter] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = nn.Parameter(torch.empty(out_f, in_f, device=self.device))
            nn.init.xavier_normal_(w)
            self.weights.append(w)

    def forward(self, x: Tensor) -> List[Tensor]:
        activations: List[Tensor] = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def build_graph(
        self,
        activations: List[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        final = activations[-1]
        n = final.size(0)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i, j in itertools.combinations(range(n), 2):
            fid = self.state_fidelity(final[i], final[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

def random_graph(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[nn.Parameter], List[Tuple[Tensor, Tensor]], Tensor]:
    net = GraphQNN(*qnn_arch)
    target_weight = net.weights[-1].detach().clone()
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(qnn_arch[0], dtype=torch.float32)
        y = target_weight @ x
        dataset.append((x, y))
    return net.arch, net.weights, dataset, target_weight

def random_graph_with_params(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[nn.Parameter], List[Tuple[Tensor, Tensor]], Tensor]:
    arch = list(qnn_arch)
    weights: List[nn.Parameter] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        w = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.xavier_normal_(w)
        weights.append(w)
    target_weight = weights[-1].detach().clone()
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(arch[0], dtype=torch.float32)
        y = target_weight @ x
        dataset.append((x, y))
    return arch, weights, dataset, target_weight

def train_step(
    net: GraphQNN,
    data: Iterable[Tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
) -> float:
    net.train()
    optimiser.zero_grad()
    losses: List[float] = []
    for x, y in data:
        activations = net.forward(x)
        pred = activations[-1]
        loss = loss_fn(pred, y)
        loss.backward()
        losses.append(loss.item())
    optimiser.step()
    return sum(losses) / len(losses)

def build_graph(
    activations: List[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    final = activations[-1]
    n = final.size(0)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j in itertools.combinations(range(n), 2):
        fid = GraphQNN.state_fidelity(final[i], final[j])
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph
