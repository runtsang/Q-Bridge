"""Hybrid classical regression model based on quantum‑inspired data
encoding, graph neural network processing, fully connected layer,
and sampler‑based expectation computation.

The module is self‑contained, uses only NumPy, PyTorch and
networkx, and can be dropped into the existing project directory.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import itertools
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

# ----------------------------------------------------------------------
# Dataset and data generation
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset exposing raw feature vectors and labels."""
    def __init__(self, samples: int = 1024, num_features: int = 10):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"features": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

# ----------------------------------------------------------------------
# Helper utilities – graph construction and feed‑forward
# ----------------------------------------------------------------------
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two normalized real vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor], threshold: float,
    *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random feed‑forward network and synthetic training data."""
    weights = [torch.randn(out, in_, dtype=torch.float32) for in_, out in zip(arch[:-1], arch[1:])]
    target_weight = weights[-1]
    training_data = [(torch.randn(arch[0], dtype=torch.float32),
                      target_weight @ torch.randn(arch[0], dtype=torch.float32)) for _ in range(samples)]
    return list(arch), weights, training_data, target_weight

def feedforward(arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Propagate a batch through a deterministic network."""
    outputs = []
    for inp, _ in samples:
        activations = [inp]
        current = inp
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        outputs.append(activations)
    return outputs

# ----------------------------------------------------------------------
# Graph processor – message passing using the random network
# ----------------------------------------------------------------------
class GraphProcessor(nn.Module):
    """Simple graph‑based feature aggregator."""
    def __init__(self, num_features: int, graph_threshold: float = 0.8):
        super().__init__()
        self.threshold = graph_threshold
        # Randomly initialise a small GNN‑style weight matrix
        self.weight = nn.Parameter(torch.randn(num_features, num_features, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Build similarity graph
        graph = fidelity_adjacency([f for f in features], self.threshold)
        # Convert graph to adjacency matrix
        adj = nx.to_numpy_array(graph)
        # Message passing: aggregate neighbour features
        aggregated = torch.from_numpy(adj @ features.numpy()).float()
        # Linear transform
        return F.relu(self.weight @ aggregated.t()).t()

# ----------------------------------------------------------------------
# Fully connected layer (inspired by FCL)
# ----------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

# ----------------------------------------------------------------------
# Sampler network – softmax output producing a probability distribution
# ----------------------------------------------------------------------
class SamplerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two outcomes."""
        return F.softmax(self.net(inputs), dim=-1)

# ----------------------------------------------------------------------
# Hybrid regression model
# ----------------------------------------------------------------------
class HybridRegression(nn.Module):
    """Classical regression model that combines data encoding,
    graph‑based propagation, fully‑connected layer, and sampler‑based
    expectation computation."""
    def __init__(
        self,
        num_features: int = 10,
        graph_threshold: float = 0.8,
        *,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.encoder = nn.Linear(num_features, 32)
        self.graph_processor = GraphProcessor(32, graph_threshold)
        self.fcl = FullyConnectedLayer()
        self.sampler = SamplerModule()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : torch.Tensor
            Raw feature matrix of shape (N, num_features).

        Returns
        -------
        torch.Tensor
            Regression target as an expectation value.
        """
        # 1. Encode raw features
        encoded = F.relu(self.encoder(batch))

        # 2. Graph‑based aggregation
        aggregated = self.graph_processor(encoded)

        # 3. Fully connected layer – produce a scalar per sample
        scalars = self.fcl(aggregated[:, 0])  # use first dimension as placeholder

        # 4. Sampler network – produce a probability distribution
        probs = self.sampler(aggregated[:, :2])  # take first two dims

        # 5. Combine: expectation = weighted sum of scalar and distribution mean
        distribution_mean = (probs * torch.tensor([0.0, 1.0], device=probs.device)).sum(dim=1)
        expectation = scalars.squeeze() * 0.5 + distribution_mean * 0.5
        return expectation

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
