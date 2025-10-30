from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor

# ----------------------------------------------------------------------
# Random linear layer generator
def _rand_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

# ----------------------------------------------------------------------
# Synthetic training data for graph embeddings
def generate_graph_training_data(
    graph: nx.Graph,
    target: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.tensor(nx.to_numpy_array(graph).flatten(), dtype=torch.float32)
        target_vec = target @ features
        data.append((features, target_vec))
    return data

# ----------------------------------------------------------------------
# Build a random graph neural network architecture
def random_graph_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = [
        _rand_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    graph = nx.erdos_renyi_graph(qnn_arch[0], 0.5)
    training_data = generate_graph_training_data(graph, target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# ----------------------------------------------------------------------
# Feedforward propagation
def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    all_acts: List[List[Tensor]] = []
    for features, _ in samples:
        acts = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            acts.append(current)
        all_acts.append(acts)
    return all_acts

# ----------------------------------------------------------------------
# Fidelity between two tensors
def tensor_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (a.norm() + 1e-12)
    b_norm = b / (b.norm() + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

# ----------------------------------------------------------------------
# Build a graph of state fidelities
def fidelity_graph(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = tensor_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----------------------------------------------------------------------
# Fast estimator utilities
class FastBaseEstimator:
    """Evaluate a neural network model on batches of inputs and scalar observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic evaluations."""
    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ----------------------------------------------------------------------
# Autoencoder for graph embeddings
class GraphAutoEncoder(nn.Module):
    """Graph autoencoder built on linear layers with ReLU and optional dropout."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Hybrid graph neural network interface
class GraphQNN(nn.Module):
    """Hybrid graph neural network that combines classical autoencoding with
    graph‑based feed‑forward propagation.  It can be trained like any PyTorch
    module and used together with the fast estimator utilities."""
    def __init__(self, architecture: Sequence[int], latent_dim: int = 32) -> None:
        super().__init__()
        input_dim = architecture[0] * architecture[0]  # flattened adjacency
        self.autoencoder = GraphAutoEncoder(input_dim, latent_dim=latent_dim)

    def forward(self, graph: nx.Graph) -> Tensor:
        features = torch.tensor(nx.to_numpy_array(graph).flatten(), dtype=torch.float32)
        return self.autoencoder(features)

__all__ = [
    "feedforward",
    "fidelity_graph",
    "random_graph_network",
    "generate_graph_training_data",
    "tensor_fidelity",
    "FastBaseEstimator",
    "FastEstimator",
    "GraphAutoEncoder",
    "GraphQNN",
]
