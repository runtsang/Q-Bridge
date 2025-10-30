"""Hybrid classical kernel method combining RBF kernels, graph neural networks, and classifier utilities."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

class HybridKernelMethod:
    """A unified interface for classical kernel computation, graph neural network utilities, 
    and classifier/sampler construction. Mirrors the quantum interface but remains purely classical."""

    # --- Kernel utilities ----------------------------------------------------
    @staticmethod
    def rbf_kernel(x: Tensor, y: Tensor, gamma: float = 1.0) -> Tensor:
        """Compute RBF kernel between two vectors."""
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        """Return Gram matrix between sequences a and b."""
        return np.array([[HybridKernelMethod.rbf_kernel(x, y, gamma).item() for y in b] for x in a])

    # --- Graph neural network utilities ---------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(HybridKernelMethod._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = HybridKernelMethod.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridKernelMethod.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --- Classifier utilities ------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    # --- Sampler utilities ---------------------------------------------------
    @staticmethod
    def SamplerQNN() -> nn.Module:
        class SamplerModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.Tanh(),
                    nn.Linear(4, 2),
                )

            def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
                return F.softmax(self.net(inputs), dim=-1)
        return SamplerModule()

    # --- Graph from kernel matrix --------------------------------------------
    @staticmethod
    def kernel_graph(a: Sequence[Tensor], threshold: float, gamma: float = 1.0) -> nx.Graph:
        """Construct a graph where nodes are samples and edges are weighted by kernel similarity."""
        mat = HybridKernelMethod.kernel_matrix(a, a, gamma)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(a)))
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                weight = mat[i, j]
                if weight >= threshold:
                    graph.add_edge(i, j, weight=weight)
        return graph

    __all__ = ["HybridKernelMethod"]
