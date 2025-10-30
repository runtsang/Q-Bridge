"""GraphQNNHybrid – classical implementation.

The class mirrors the original GraphQNN module but adds:
  * a fully‑connected layer inspired by the FCL example,
  * a quanvolution filter for image data,
  * a classifier builder that follows the quantum analogue,
  * fidelity‑based graph construction.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

Tensor = torch.Tensor


class FullyConnectedLayer(nn.Module):
    """Simple linear layer with tanh activation – FCL analogue."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach().numpy()


class QuanvolutionFilter(nn.Module):
    """2×2 image patch filter using a 2‑qubit random unitary."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that chains a QuanvolutionFilter with a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class GraphQNNHybrid:
    """Classical graph‑based neural network with quantum‑inspired layers."""

    def __init__(self, qnn_arch: Sequence[int], device: str | torch.device = "cpu") -> None:
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.network = self._build_network()
        self.network.to(self.device)

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.extend([nn.Linear(in_f, out_f), nn.ReLU()])
        layers.append(nn.Linear(self.arch[-1], 1))
        return nn.Sequential(*layers)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        activations: List[List[Tensor]] = []
        for features, _ in samples:
            cur = features.to(self.device)
            layerwise = [cur]
            for layer in self.network:
                cur = torch.tanh(layer(cur))
                layerwise.append(cur)
            activations.append(layerwise)
        return activations

    @staticmethod
    def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            feat = torch.randn(target.shape[0], dtype=torch.float32)
            tgt = target @ feat
            dataset.append((feat, tgt))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor], threshold: float,
        *, secondary: float | None = None,
        secondary_weight: float = 0.5
    ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(sa, sb)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Construct a classical classifier mirroring the quantum version."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            layers.extend([lin, nn.ReLU()])
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        net = nn.Sequential(*layers)
        observables = list(range(2))
        return net, encoding, weight_sizes, observables


__all__ = [
    "FullyConnectedLayer",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "GraphQNNHybrid",
]
