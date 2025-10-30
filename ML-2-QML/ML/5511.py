from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from.QCNN import QCNNModel
from.GraphQNN import state_fidelity
from.SelfAttention import SelfAttention

class HybridQCNNGraphAttention(nn.Module):
    """Classical hybrid QCNN with graph diffusion and self‑attention.

    The model combines:
    * a graph neural network that diffuses features along a fidelity‑based
      adjacency graph,
    * a learnable self‑attention block (adapted from the classical
      SelfAttention helper),
    * the original QCNNModel which implements a stack of fully‑connected
      layers mimicking the quantum convolution steps.
    """
    def __init__(self, embed_dim: int = 8, threshold: float = 0.8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold

        # learnable attention parameters
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # simple graph neural network layer
        self.gnn = nn.Linear(embed_dim, embed_dim)

        # QCNN stack
        self.qcnn = QCNNModel()

    def _construct_adjacency(self, batch: torch.Tensor) -> torch.Tensor:
        """Build a batch‑wise adjacency matrix from pairwise fidelities."""
        batch = F.normalize(batch, dim=1)
        n = batch.size(0)
        adj = torch.zeros(n, n, device=batch.device)
        for i in range(n):
            for j in range(i + 1, n):
                fid = state_fidelity(batch[i], batch[j])
                if fid >= self.threshold:
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    def _graph_diffusion(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Weighted sum of neighbours followed by a linear transform."""
        neighbour_sum = adj @ features
        return F.relu(self.gnn(neighbour_sum))

    def _self_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Apply a learnable self‑attention block."""
        sa = SelfAttention()
        # Override the embed_dim to match the feature dimensionality
        sa.embed_dim = self.embed_dim
        rot = self.rotation.detach().cpu().numpy()
        ent = self.entangle.detach().cpu().numpy()
        return torch.tensor(sa.run(rot, ent, features.detach().cpu().numpy()),
                            device=features.device, dtype=features.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through adjacency, attention and QCNN."""
        adj = self._construct_adjacency(inputs)
        diffused = self._graph_diffusion(inputs, adj)
        attended = self._self_attention(diffused)
        return self.qcnn(attended)

def QCNN() -> HybridQCNNGraphAttention:
    """Factory returning the hybrid QCNN with graph and attention."""
    return HybridQCNNGraphAttention()
