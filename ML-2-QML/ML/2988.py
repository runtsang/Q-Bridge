from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor

class GraphQNNGen256(nn.Module):
    """Hybrid classical graph neural network that fuses CNN feature extraction
    with graph‑based propagation.  The architecture is built from an arbitrary
    layer list (default 256 nodes) and can be used as a drop‑in replacement
    for the original GraphQNN while providing richer feature extraction."""
    def __init__(
        self,
        arch: Sequence[int] = (256,),
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        # CNN encoder mimicking Quantum‑NAT
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Graph‑layer weights
        self.graph_layers = nn.ModuleList(
            [nn.Linear(in_f, out_f, bias=False)
             for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )
        self.norm = nn.BatchNorm1d(self.arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        """Encode, propagate through graph layers and normalize."""
        # CNN feature extraction
        feats = self.encoder(x)
        feats = feats.view(feats.size(0), -1)
        # Graph propagation
        h = feats
        for layer in self.graph_layers:
            h = torch.tanh(layer(h))
        return self.norm(h)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(self, states: Sequence[Tensor]) -> nx.Graph:
        """Build weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and fid >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run a batch of samples through the network, recording all layer
        activations."""
        stored: List[List[Tensor]] = []
        for inp, _ in samples:
            h = inp
            activations = [h]
            for layer in self.graph_layers:
                h = torch.tanh(layer(h))
                activations.append(h)
            stored.append(activations)
        return stored

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network with weights drawn from N(0,1) and
        synthetic training data based on the last layer."""
        weights = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target = weights[-1]
        training_data = self.random_training_data(target, samples)
        return self.arch, weights, training_data, target

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Synthetic data where targets are linear transformations of inputs."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

__all__ = ["GraphQNNGen256"]
