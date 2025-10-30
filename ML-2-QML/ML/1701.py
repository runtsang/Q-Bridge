from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class SharedGraphQNN:
    """Hybrid classical graph neural network.

    Parameters
    ----------
    architecture : Sequence[int]
        Number of nodes per layer. The first element is the input size.
    device : str, optional
        Target device for tensors.
    """

    def __init__(self, architecture: Sequence[int], device: str = "cpu"):
        self.architecture = list(architecture)
        self.device = torch.device(device)
        self.layers = self._build_layers()
        self.to(self.device)

    def _build_layers(self) -> List[nn.Module]:
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        return layers

    def to(self, device: torch.device) -> "SharedGraphQNN":
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations for each layer."""
        activations: List[torch.Tensor] = [x]
        current = x
        for layer in self.layers:
            current = layer(current)
            activations.append(current)
        return activations

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap between two normalized vectors."""
        a_norm = a / (a.norm() + 1e-12)
        b_norm = b / (b.norm() + 1e-12)
        return float((a_norm @ b_norm).abs() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g

    @staticmethod
    def random_training_data(
        target: torch.Tensor,
        samples: int,
        noise: float = 0.0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            x = torch.randn(target.shape[1], device=target.device)
            if noise > 0:
                x = x + noise * torch.randn_like(x)
            y = target @ x
            dataset.append((x, y))
        return dataset

    @staticmethod
    def random_network(
        architecture: Sequence[int],
        samples: int = 32,
    ) -> Tuple[List[int], List[nn.Module], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Dummy target linear layer for generating training data
        target = nn.Linear(architecture[-2], architecture[-1])
        target.weight.data = torch.randn_like(target.weight)
        target.bias.data = torch.randn_like(target.bias)
        target_tensor = torch.cat([target.weight, target.bias.unsqueeze(1)], dim=1)
        training_data = SharedGraphQNN.random_training_data(target_tensor, samples)
        layers = [target]
        return list(architecture), layers, training_data, target_tensor

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for x, y in dataset:
                optimizer.zero_grad()
                out = self.forward(x)[-1]
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
