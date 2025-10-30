"""
HybridQuantumCNN: A classical network that emulates a quantum convolutional neural network,
leverages graph-based adjacency for feature aggregation, and incorporates a transformer
block for sequence modeling. The architecture is inspired by QCNN, GraphQNN, and
classical transformer designs, providing a robust baseline for hybrid quantum-classical
experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List

# Fidelity utilities (from GraphQNN)
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: List[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j in range(i + 1, len(states)):
            state_j = states[j]
            fid = state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Simple graph convolution layer
class GraphConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # adj: (batch, N, N), x: (batch, N, F)
        adj_norm = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(adj_norm, x)

# Classical transformer block
class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

# Hybrid Quantum CNN (classical implementation)
class HybridQuantumCNN(nn.Module):
    """
    A hybrid classical network that mirrors the structure of a QCNN, augments it with
    graph-based adjacency for feature aggregation, and finishes with a transformer
    block for sequence modeling. This design serves as a robust baseline for
    later quantum enhancement.
    """

    def __init__(
        self,
        num_classes: int = 2,
        transformer_blocks: int = 2,
        ffn_dim: int = 64,
    ) -> None:
        super().__init__()
        # Feature map (8 -> 16)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Graph convolution (optional)
        self.graph_conv = GraphConv()
        # Transformer block
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(4, num_heads=2, ffn_dim=ffn_dim)
                for _ in range(transformer_blocks)
            ]
        )
        # Classifier
        self.classifier = nn.Linear(4, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 8)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)  # (batch, 4)
        # Prepare for transformer: sequence length 1
        seq = x.unsqueeze(1)  # (batch, 1, 4)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        out = self.classifier(seq)
        return torch.sigmoid(out) if out.shape[-1] == 1 else out

__all__ = ["HybridQuantumCNN"]
