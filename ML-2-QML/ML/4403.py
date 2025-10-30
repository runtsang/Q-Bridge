"""Hybrid classical Graph QNN with autoencoder and QCNN emulation.

This module extends the original GraphQNN utilities by adding:
- Random graph neural network generation with autoencoder-based feature extraction.
- Classical feedforward using torch.
- Fidelity-based adjacency graph construction.
- QCNNModel emulation for quantum-inspired convolution.
"""

import itertools
from typing import Iterable, Sequence, Tuple, List, Optional
import networkx as nx
import torch
import numpy as np

from.Autoencoder import Autoencoder, AutoencoderConfig
from.QCNN import QCNNModel

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


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


class GraphQNN:
    """Hybrid classical graph neural network with autoencoder support."""
    def __init__(self, arch: Sequence[int], autoencoder_cfg: Optional[AutoencoderConfig] = None):
        self.arch = list(arch)
        self.autoencoder = Autoencoder(autoencoder_cfg.input_dim) if autoencoder_cfg else None
        self.weights: List[Tensor] = []
        self.training_data: List[Tuple[Tensor, Tensor]] = []
        self.target: Tensor = None

    def build_random(self, samples: int):
        self.arch, self.weights, self.training_data, self.target = random_network(self.arch, samples)

    def encode_graph(self, graph: nx.Graph) -> Tensor:
        """Encode adjacency matrix of graph into latent vector via autoencoder."""
        if not self.autoencoder:
            raise RuntimeError("Autoencoder not configured.")
        adj = nx.to_numpy_array(graph)
        flat = torch.from_numpy(adj).float()
        with torch.no_grad():
            return self.autoencoder.encode(flat)

    def forward(self, inputs: Tensor) -> List[Tensor]:
        """Classical feedforward."""
        return feedforward(self.arch, self.weights, [(inputs, None)])

    def hybrid_forward(self, graph: nx.Graph) -> List[Tensor]:
        """Encode graph, then feedforward."""
        latent = self.encode_graph(graph)
        return self.forward(latent)

    def train(self, epochs: int = 50, lr: float = 1e-3):
        """Simple training loop for the classical network."""
        if not self.weights:
            raise RuntimeError("Model weights not initialized.")
        optimizer = torch.optim.Adam(self.weights, lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = [self.forward(x)[-1] for x, _ in self.training_data]
            loss = loss_fn(torch.stack(outputs), torch.stack([t for _, t in self.training_data]))
            loss.backward()
            optimizer.step()

    @staticmethod
    def QCNN() -> QCNNModel:
        """Return a QCNNModel instance for quantum-inspired convolution."""
        return QCNNModel()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
