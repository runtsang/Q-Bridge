import torch
import torch.nn.functional as F
from torch import nn
import networkx as nx
import numpy as np
from typing import Optional

class ConvGraphQNN(nn.Module):
    """
    Classical hybrid convolution + graph neural network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        graph_threshold: float = 0.9,
        secondary_threshold: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

        # Linear layer mimicking a convolution filter over a patch
        self.linear = nn.Linear(kernel_size * kernel_size, 1, bias=True)

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slide a kernel over the image and return a scalar per patch.
        """
        B, C, H, W = x.shape
        unfold = F.unfold(x, kernel_size=self.kernel_size, stride=1)
        patches = unfold.permute(0, 2, 1)  # [B, L, C*k*k]
        logits = self.linear(patches)  # [B, L, 1]
        activations = torch.sigmoid(logits.squeeze(-1) - self.conv_threshold)
        return activations  # [B, L]

    def _build_fidelity_graph(self, feats: torch.Tensor) -> nx.Graph:
        """
        Construct a weighted graph from feature vectors using cosine similarity.
        """
        if feats.dim() == 3:
            feats = feats.mean(dim=0)  # aggregate batch
        n = feats.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        norms = torch.norm(feats, dim=1, keepdim=True) + 1e-12
        normalized = feats / norms
        sim = normalized @ normalized.t()  # [n, n]
        for i in range(n):
            for j in range(i + 1, n):
                fid = sim[i, j].item()
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif self.secondary_threshold is not None and fid >= self.secondary_threshold:
                    graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    def _graph_propagate(self, feats: torch.Tensor, graph: nx.Graph) -> torch.Tensor:
        """
        Simple weighted aggregation of neighboring features.
        """
        n = feats.shape[0]
        out = feats.clone()
        for node in graph.nodes:
            neigh = list(graph.neighbors(node))
            if neigh:
                weights = torch.tensor(
                    [graph[node][nbr]["weight"] for nbr in neigh],
                    dtype=torch.float32,
                    device=feats.device,
                )
                neigh_feats = feats[neigh]
                weighted_sum = torch.sum(neigh_feats * weights[:, None], dim=0)
                out[node] += weighted_sum / weights.sum()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        """
        B = x.shape[0]
        patches = self._patch_features(x)  # [B, L]
        feats = patches.unsqueeze(-1)  # [B, L, 1]
        outputs = []
        for b in range(B):
            graph = self._build_fidelity_graph(feats[b].squeeze(-1))
            propagated = self._graph_propagate(feats[b].squeeze(-1), graph)
            outputs.append(propagated.mean(dim=0))
        return torch.stack(outputs)

__all__ = ["ConvGraphQNN"]
