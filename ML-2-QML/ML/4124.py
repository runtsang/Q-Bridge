from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


class GraphQNNGen099ML:
    """Classical graph neural network utilities with convolution and classifier support."""

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> list[tuple[Tensor, Tensor]]:
        dataset: list[tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: list[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNNGen099ML._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen099ML.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[tuple[Tensor, Tensor]],
    ) -> list[list[Tensor]]:
        stored: list[list[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen099ML.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
        """Return a PyTorch convolution filter emulating a quanvolution layer."""
        class ConvFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.threshold = threshold
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

            def run(self, data) -> float:
                tensor = torch.as_tensor(data, dtype=torch.float32)
                tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
                logits = self.conv(tensor)
                activations = torch.sigmoid(logits - self.threshold)
                return activations.mean().item()

        return ConvFilter()

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int):
        layers: list[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: list[int] = []
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


__all__ = [
    "GraphQNNGen099ML",
]
