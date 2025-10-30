from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import networkx as nx
from typing import Sequence

# ---------------------------------------------------------------------------#
#  Data generation and dataset
# ---------------------------------------------------------------------------#
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    The target is a smooth function of the sum of the input features,
    mimicking a superposition‑like relationship.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) + 0.1 * np.cos(2 * angles)).astype(np.float32)
    return x, y

class RegressionDataset(Dataset):
    """Torch dataset that yields a single feature vector and a scalar target."""
    __slots__ = ("features", "labels")

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ---------------------------------------------------------------------------#
#  Classical RBF kernel utilities
# ---------------------------------------------------------------------------#
class _RBFAnsatz(nn.Module):
    """Computes the radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a forward signature compatible with the ML seed."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = _RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # broadcast to (1, d)
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Return the Gram matrix between two sets of feature vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------------------------------------------------------------------------#
#  Graph utilities (diagnostic only)
# ---------------------------------------------------------------------------#
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---------------------------------------------------------------------------#
#  Classical regression models
# ---------------------------------------------------------------------------#
class KRRModel(nn.Module):
    """Kernel ridge regression using the RBF kernel."""
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_vec = None  # to be set by ``fit``

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Compute dual coefficients α = (K + αI)⁻¹ y."""
        K = torch.tensor(kernel_matrix(X, X, self.gamma), dtype=torch.float32)
        self.alpha_vec = torch.linalg.solve(K + self.alpha * torch.eye(K.shape[0]), y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.alpha_vec is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = torch.tensor(kernel_matrix(X, X, self.gamma), dtype=torch.float32)
        return torch.mv(K, self.alpha_vec)

class HybridKModel(nn.Module):
    """Concatenates classical RBF features with optional quantum embeddings."""
    def __init__(self, num_features: int, num_centers: int = 50, gamma: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.num_centers = num_centers
        self.gamma = gamma
        self.register_buffer("centers", torch.rand(num_centers, num_features))
        self.head = nn.Linear(num_centers, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RBF kernel between input and centres
        feats = torch.exp(-self.gamma * torch.sum((x.unsqueeze(1) - self.centers) ** 2, dim=-1))
        return self.head(feats).squeeze(-1)

# ---------------------------------------------------------------------------#
#  Legacy fully‑connected model (for backward compatibility)
# ---------------------------------------------------------------------------#
class QModel(nn.Module):
    """Legacy fully‑connected regression model from the seed."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

__all__ = [
    "RegressionDataset",
    "generate_superposition_data",
    "Kernel",
    "kernel_matrix",
    "state_fidelity",
    "fidelity_adjacency",
    "KRRModel",
    "HybridKModel",
    "QModel",
]
