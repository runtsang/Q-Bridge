"""Classical feature graph network for the hybrid QCNN architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List

Tensor = torch.Tensor

class FeatureGraphNet(nn.Module):
    """
    Maps an 8‑dimensional input to an 8‑node weighted adjacency matrix.
    The output is row‑wise softmaxed to form a stochastic matrix that can be
    interpreted as a learned graph.
    """
    def __init__(self, in_dim: int = 8, hidden_dim: int = 32, out_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim * out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        logits = self.net(x).view(batch, self.out_dim, self.out_dim)
        return F.softmax(logits, dim=-1)

def adjacency_to_edge_list(adj: Tensor, threshold: float = 0.2) -> List[Tuple[int, int]]:
    """
    Convert a batch‑size‑1 adjacency tensor to a list of edges.
    """
    adj_np = adj.squeeze(0).detach().cpu().numpy()
    n = adj_np.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if adj_np[i, j] > threshold and i!= j:
                edges.append((i, j))
    return edges

class QuantumHybridCNN(nn.Module):
    """
    Wrapper that holds the classical graph generator.  The quantum part is
    constructed separately (see the QML module).  This class exposes a
    ``forward`` method that returns the adjacency matrix for a given input.
    """
    def __init__(self, in_dim: int = 8, hidden_dim: int = 32, out_dim: int = 8) -> None:
        super().__init__()
        self.graph_net = FeatureGraphNet(in_dim, hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.graph_net(x)

__all__ = ["FeatureGraphNet", "adjacency_to_edge_list", "QuantumHybridCNN"]
