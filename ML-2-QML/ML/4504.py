"""Hybrid classical self‑attention module combining attention, regression, graph, and sampler."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# 1. Regression utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic data for a simple regression task."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """Simple feed‑forward network for regression."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# 2. Graph utilities
# --------------------------------------------------------------------------- #

def fidelity_adjacency(states: torch.Tensor,
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """
    Build a weighted graph where edges represent fidelity between state vectors.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = torch.dot(states[i], states[j]) / (
                torch.norm(states[i]) * torch.norm(states[j])
            )
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 3. Sampler network
# --------------------------------------------------------------------------- #

class SamplerModule(nn.Module):
    """A tiny neural sampler that outputs a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------------- #
# 4. Hybrid self‑attention class
# --------------------------------------------------------------------------- #

class HybridSelfAttention:
    """
    Classical hybrid self‑attention that can optionally combine a regression head,
    a graph‑based adjacency weighting, and a lightweight sampler network.
    """

    def __init__(self,
                 embed_dim: int,
                 graph_threshold: float = 0.8,
                 use_sampler: bool = False):
        self.embed_dim = embed_dim
        self.graph_threshold = graph_threshold
        self.use_sampler = use_sampler

        # Classical attention parameters
        self.rotation_params = torch.randn(embed_dim, embed_dim, dtype=torch.float32)
        self.entangle_params = torch.randn(embed_dim, embed_dim, dtype=torch.float32)

        # Regression components
        self.reg_model = RegressionModel(embed_dim)
        self.reg_dataset = RegressionDataset(samples=200, num_features=embed_dim)
        self.reg_loader = DataLoader(self.reg_dataset, batch_size=32)

        # Sampler
        self.sampler = SamplerModule() if use_sampler else None

    def run_classical_attention(self, inputs: np.ndarray) -> np.ndarray:
        """Standard scaled dot‑product self‑attention."""
        q = torch.tensor(inputs @ self.rotation_params, dtype=torch.float32)
        k = torch.tensor(inputs @ self.entangle_params, dtype=torch.float32)
        v = torch.tensor(inputs, dtype=torch.float32)
        scores = torch.softmax((q @ k.T) / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()

    def build_graph(self, states: np.ndarray) -> nx.Graph:
        """Build a fidelity‑based adjacency graph from raw state vectors."""
        state_tensors = torch.tensor(states, dtype=torch.float32)
        return fidelity_adjacency(state_tensors, self.graph_threshold)

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """If a sampler is present, produce a probability distribution."""
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled in this instance.")
        inp = torch.tensor(inputs, dtype=torch.float32)
        return self.sampler(inp).detach().numpy()

    def train_regression(self, epochs: int = 5, lr: float = 1e-3) -> None:
        """Simple training loop for the regression head."""
        optimizer = torch.optim.Adam(self.reg_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for batch in self.reg_loader:
                features = batch["states"]
                target = batch["target"]
                pred = self.reg_model(features)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
