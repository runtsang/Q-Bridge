import torch
from torch import nn
import numpy as np
import networkx as nx
from typing import List, Tuple, Iterable, Sequence


class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter that mimics a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class QCNNHybridModel(nn.Module):
    """Classical QCNN‑style network with convolution, pooling, and graph utilities."""
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: Sequence[int] = (16, 12, 8, 4),
                 kernel_size: int = 2,
                 threshold: float = 0.0):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh()
        )
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            self.conv_layers.append(nn.Sequential(nn.Linear(prev, h), nn.Tanh()))
            self.pool_layers.append(nn.Sequential(nn.Linear(h, h // 2), nn.Tanh()))
            prev = h // 2
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return torch.sigmoid(self.head(x))

    # ------------------------------------------------------------------
    # graph and kernel utilities
    # ------------------------------------------------------------------
    def compute_fidelity_graph(self,
                               outputs: Iterable[torch.Tensor],
                               threshold: float,
                               secondary: float | None = None,
                               secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities of output states."""
        graph = nx.Graph()
        states = [o.detach().cpu().numpy() for o in outputs]
        n = len(states)
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                fid = np.dot(states[i], states[j]) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gamma: float = 1.0) -> np.ndarray:
        """Compute an RBF kernel matrix between two collections of vectors."""
        A = np.vstack([x.numpy() for x in a])
        B = np.vstack([x.numpy() for x in b])
        diff = np.expand_dims(A, 1) - np.expand_dims(B, 0)
        return np.exp(-gamma * np.sum(diff ** 2, axis=2))

    # ------------------------------------------------------------------
    # data generation utilities
    # ------------------------------------------------------------------
    def random_training_data(self, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic linear data for supervised learning."""
        weight = torch.randn(self.head.out_features, self.head.in_features)
        data = []
        for _ in range(samples):
            features = torch.randn(self.head.in_features)
            target = weight @ features
            data.append((features, target))
        return data


def QCNNHybrid() -> QCNNHybridModel:
    """Factory that returns a fully configured QCNNHybridModel."""
    return QCNNHybridModel()


__all__ = ["QCNNHybrid", "QCNNHybridModel", "ConvFilter"]
