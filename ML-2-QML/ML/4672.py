import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Tuple
import numpy as np
import networkx as nx
import itertools

__all__ = ["ConvGen118"]

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return squared overlap between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create adjacency graph from a list of state tensors."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class ConvGen118(nn.Module):
    """Unified hybrid conv‑like module combining classical convolution,
    dropout, a simple fully‑connected layer, and graph adjacency construction.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 dropout_prob: float = 0.1,
                 graph_threshold: float = 0.9,
                 secondary_threshold: float | None = None,
                 secondary_weight: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.dropout = nn.Dropout(dropout_prob)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.fc = nn.Linear(1, 1)  # placeholder fully‑connected
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution, threshold, dropout, and return mean activation."""
        if data.ndim == 3:
            data = data.unsqueeze(0)
        conv_out = self.conv(data)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        conv_out = self.dropout(conv_out)
        return conv_out.mean()

    def run_fcl(self, thetas: Iterable[float]) -> torch.Tensor:
        """Apply the fully‑connected layer to a sequence of thetas."""
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(1)
        out = torch.tanh(self.fc(theta_tensor)).mean()
        return out

    def compute_graph(self, features: List[torch.Tensor]) -> nx.Graph:
        """Construct a graph from feature vectors based on fidelity thresholds."""
        return fidelity_adjacency(features,
                                  threshold=self.graph_threshold,
                                  secondary=self.secondary_threshold,
                                  secondary_weight=self.secondary_weight)
