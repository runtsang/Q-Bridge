from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import Dataset

# --------------------------------------------------------------------
# Data generation – mirrors the quantum superposition example.
# --------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a toy regression dataset where the target is a nonlinear
    function of the sum of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------
# Graph utilities – adapted from GraphQNN.py
# --------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int):
    """
    Generate synthetic training pairs by applying a linear map
    to random feature vectors.
    """
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    """
    Construct a random feed‑forward archive consisting of linear
    layers whose weights are sampled from a normal distribution.
    """
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: list[int], weights: list[torch.Tensor], samples: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Forward pass through the random network – returns all layer activations.
    """
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Absolute squared overlap between two real vectors.
    """
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """
    Build a weighted adjacency graph from state fidelities.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------
# Hybrid estimator – combines classical feed‑forward regression
# with a graph‑based regulariser.
# --------------------------------------------------------------------
class EstimatorQNN(nn.Module):
    """
    Classical regression network that embeds a graph‑based Laplacian
    regularisation on the hidden representation.  The network
    architecture mirrors the simple feed‑forward model in
    EstimatorQNN.py but augments it with a latent graph that
    captures similarity between examples.
    """
    def __init__(self, num_features: int,
                 latent_dim: int = 32,
                 graph_threshold: float = 0.8) -> None:
        super().__init__()
        self.graph_threshold = graph_threshold
        self.encoder = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        out = self.head(latent).squeeze(-1)
        return out

    def regularization(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute a Laplacian regularisation term from the fidelity
        adjacency of the latent vectors.
        """
        latent_np = latent.detach().cpu().numpy()
        graph = fidelity_adjacency(
            [torch.tensor(v, dtype=torch.float32) for v in latent_np],
            self.graph_threshold
        )
        # Laplacian matrix
        L = nx.laplacian_matrix(graph).astype(float).todense()
        # Quadratic form: 0.5 * sum_i,j L_ij * <x_i, x_j>
        latent_flat = latent.view(latent.size(0), -1)
        L_torch = torch.from_numpy(L).to(latent_flat.device).float()
        reg = 0.5 * torch.sum((latent_flat @ latent_flat.T) * L_torch)
        return reg

# --------------------------------------------------------------------
# Simple sampler – mirrors the SamplerQNN in SamplerQNN.py
# --------------------------------------------------------------------
class SamplerQNN(nn.Module):
    """
    Softmax classifier that can be used as a drop‑in replacement
    for the quantum sampler in the original example.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

__all__ = [
    "EstimatorQNN",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "random_training_data",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
