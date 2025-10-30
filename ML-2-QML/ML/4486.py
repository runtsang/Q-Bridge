import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List, Iterable

class QuanvolutionFilter(nn.Module):
    """Two‑by‑two quantum‑style convolutional filter using a standard conv layer."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class QuantumClassifierModel(nn.Module):
    """
    Classical hybrid classifier that optionally uses a Quanvolution front‑end and builds
    a fidelity graph of activations for graph‑based analysis.
    """
    def __init__(self, num_features: int, depth: int, num_classes: int = 2, use_quanvolution: bool = False):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.qfilter = QuanvolutionFilter()
            # Rough estimate of feature dimension after convolution
            feature_dim = 4 * (num_features // 2) ** 2
        else:
            feature_dim = num_features

        # Build a simple feed‑forward backbone
        layers: List[nn.Module] = []
        in_dim = feature_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, num_classes))
        self.backbone = nn.Sequential(*layers)

        self.fidelity_graph: nx.Graph | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.qfilter(x)
        activations = [x]
        current = x
        for layer in self.backbone:
            current = layer(current)
            activations.append(current)
        # Store fidelity graph of hidden states
        self.fidelity_graph = self._build_fidelity_graph(activations)
        return current

    def _build_fidelity_graph(self, activations: List[torch.Tensor]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for i, a_i in enumerate(activations):
            for j in range(i + 1, len(activations)):
                fid = torch.dot(a_i, a_j) / (torch.norm(a_i) * torch.norm(a_j) + 1e-12)
                if fid > 0.8:
                    graph.add_edge(i, j, weight=float(fid))
        return graph

    def get_fidelity_graph(self) -> nx.Graph | None:
        """Return the last computed fidelity graph or None if not yet computed."""
        return self.fidelity_graph

__all__ = ["QuantumClassifierModel"]
