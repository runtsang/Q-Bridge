import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import networkx as nx

class ClassicalSelfAttention(nn.Module):
    """Multi‑head self‑attention with residual connection."""
    def __init__(self, embed_dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.size()
        q = self.W_q(x).view(batch, seq_len, self.heads, self.d_k).transpose(1,2)
        k = self.W_k(x).view(batch, seq_len, self.heads, self.d_k).transpose(1,2)
        v = self.W_v(x).view(batch, seq_len, self.heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.W_o(out)
        return out + x  # residual

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that mimics the patch‑wise quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid classical model combining a convolution, self‑attention, and a linear head."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 32, heads: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attn = ClassicalSelfAttention(embed_dim=embed_dim, heads=heads)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        feat = self.qfilter(x)                     # -> (batch, 8, 28, 28)
        feat = F.avg_pool2d(feat, kernel_size=2)   # -> (batch, 8, 14, 14)
        feat = feat.view(feat.size(0), -1, feat.size(1))  # -> (batch, 14*14, 8)
        feat = self.attn(feat)                     # -> (batch, 14*14, 8)
        pooled = feat.mean(dim=1)                  # -> (batch, 8)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return F.log_softmax(logits, dim=-1)

def fidelity_adjacency(
    states: list[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    """Construct a weighted adjacency graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = (a @ b).abs().item() ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "ClassicalSelfAttention",
    "QuanvolutionFilter",
    "HybridQuanvolutionClassifier",
    "fidelity_adjacency",
]
