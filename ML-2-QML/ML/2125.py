import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GraphQNN__gen343(nn.Module):
    # Classical graph neural network with optional quantum-inspired layers.
    def __init__(self, qnn_arch: Sequence[int], use_quantum: bool = False, device: str = "cpu"):
        super().__init__()
        self.arch = list(qnn_arch)
        self.use_quantum = use_quantum
        self.device = device
        self.weights = nn.ParameterList(self._init_weights())

    def _init_weights(self) -> List[nn.Parameter]:
        weights = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            if self.use_quantum:
                # Random orthogonal matrix as quantum-inspired layer
                mat = torch.randn(out_f, in_f, dtype=torch.float32, device=self.device)
                q, _ = torch.qr(mat)
                weight = nn.Parameter(q)
            else:
                weight = nn.Parameter(torch.randn(out_f, in_f, dtype=torch.float32, device=self.device))
            weights.append(weight)
        return weights

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int, use_quantum: bool = False):
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            if use_quantum:
                mat = torch.randn(out_f, in_f)
                q, _ = torch.qr(mat)
                weight = q
            else:
                weight = torch.randn(out_f, in_f)
            weights.append(weight)
        target_weight = weights[-1]
        training_data = GraphQNN__gen343.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for weight in self.weights[:-1]:
            out = torch.tanh(weight @ out.t()).t()
        out = self.weights[-1] @ out.t()
        return out.t()

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current.t()).t()
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen343.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        fidelity_reg: float = 0.0,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            total_loss = 0.0
            for features, target in dataset:
                features = features.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(features)
                mse = F.mse_loss(output, target)
                loss = mse
                if fidelity_reg > 0.0:
                    fid = self.state_fidelity(output, target)
                    loss += fidelity_reg * (1.0 - fid)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return
