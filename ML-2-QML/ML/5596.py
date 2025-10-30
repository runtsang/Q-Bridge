import torch
from torch import nn
import torch.nn.functional as F
import networkx as nx
import itertools

class QCNNGen504Model(nn.Module):
    """
    Classical hybrid network that emulates a QCNN with a regression head
    and exposes graph‑based state similarity diagnostics.
    """
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.Tanh())
        # Convolution‑like layers
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dims[i-1], hidden_dims[i]), nn.Tanh()))
        # Regression head
        self.reg_head = nn.Linear(hidden_dims[-1], 1)
        # Sampler network for categorical output
        self.sampler = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = [x]
        h = self.feature_map(x)
        activations.append(h)
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        out = torch.sigmoid(self.reg_head(h))
        return out, activations

    def compute_adjacency(
        self,
        activations: list[torch.Tensor],
        threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from pairwise cosine similarities of activations.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for (i, a), (j, b) in itertools.combinations(enumerate(activations), 2):
            sim = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
            fid = sim.item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return categorical distribution from the sampler network.
        """
        return F.softmax(self.sampler(inputs), dim=-1)

__all__ = ["QCNNGen504Model"]
