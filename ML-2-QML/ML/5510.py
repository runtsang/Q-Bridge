import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic regression data where the target depends on a nonlinear
    combination of input features.  The data distribution mimics the
    superposition states used in the quantum counterpart.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with feature tensor and target scalar.
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

class QuantumRegression(nn.Module):
    """
    Classical regression model that augments a feed‑forward network with
    graph‑convolutional aggregation based on pairwise cosine similarity.
    """
    def __init__(self, num_features: int, graph_threshold: float = 0.8):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.graph_threshold = graph_threshold
        self.graph_conv = nn.Linear(32, 32)
        self.head = nn.Linear(32, 1)

    def _fidelity_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """
        Build a binary adjacency matrix from cosine similarity of hidden
        representations.  Self‑loops are added to preserve each node's
        contribution.
        """
        normed = F.normalize(features, dim=1)
        sim = torch.mm(normed, normed.t())
        adj = (sim >= self.graph_threshold).float()
        adj.fill_diagonal_(1.0)
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        adj = self._fidelity_adjacency(feats)
        agg = torch.matmul(adj, feats)          # neighborhood aggregation
        agg = self.graph_conv(agg)
        out = self.head(agg).squeeze(-1)
        return out

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
