import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


class Kernel(nn.Module):
    """Classical RBF kernel used as a surrogate for the quantum kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class GraphNN(nn.Module):
    """Simple graph neural network that aggregates features via a fixed adjacency matrix."""
    def __init__(self, arch: list[int], adjacency: nx.Graph):
        super().__init__()
        self.arch = arch
        self.linear_layers = nn.ModuleList([nn.Linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1)])
        self.adj_matrix = torch.tensor(nx.to_numpy_array(adjacency), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.linear_layers:
            h = torch.matmul(self.adj_matrix, h)
            h = torch.tanh(linear(h))
        return h


class QuanvolutionGen196(nn.Module):
    """Hybrid network that merges classical convolution, a quantum‑style kernel, and a graph‑based layer."""
    def __init__(
        self,
        conv_out: int = 4,
        prototype_count: int = 10,
        kernel_gamma: float = 1.0,
        adj_threshold: float = 0.8,
        graph_arch: list[int] = [4, 4, 10]
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_out, kernel_size=2, stride=2)
        self.kernel = Kernel(gamma=kernel_gamma)

        # Prototype vectors for kernel similarity and graph construction
        self.prototypes = nn.Parameter(torch.randn(prototype_count, conv_out * 14 * 14))

        # Build prototype adjacency once (offline)
        with torch.no_grad():
            prot_mat = self.prototypes.unsqueeze(0)      # 1 x P x D
            prot_mat_t = self.prototypes.unsqueeze(1)   # P x 1 x D
            sim = torch.exp(-kernel_gamma * torch.sum((prot_mat - prot_mat_t) ** 2, dim=-1))
            adj = (sim >= adj_threshold).float()
        self.graph = GraphNN(graph_arch, nx.from_numpy_array(adj.numpy()))

        self.final = nn.Linear(graph_arch[-1], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical convolution
        conv_feat = self.conv(x)                       # B x conv_out x 14 x 14
        conv_flat = conv_feat.view(x.size(0), -1)      # B x D

        # Kernel similarity to prototypes
        B = x.size(0)
        P = self.prototypes.size(0)
        k_mat = torch.zeros(B, P, device=x.device)
        for i in range(B):
            for j in range(P):
                k_mat[i, j] = self.kernel(conv_flat[i], self.prototypes[j])

        # Aggregate prototype‑weighted features
        features = torch.matmul(k_mat, self.prototypes)  # B x D

        # Graph propagation
        graph_out = self.graph(features)

        logits = self.final(graph_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Kernel", "GraphNN", "QuanvolutionGen196"]
