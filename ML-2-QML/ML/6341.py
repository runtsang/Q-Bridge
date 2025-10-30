"""HybridGraphNet: classical baseline for GraphQNN and QCNet.

The module defines a single PyTorch nn.Module that
* concatenates image features (via a small CNN) with graph embeddings
  produced by a deterministic graph neural network (GNN).
* replaces the quantum expectation head with a sigmoid‑like
  differentiable function (HybridFunction) that mimics the
  original quantum layer.
* exposes a ``Hybrid`` class that is a drop‑in replacement for the
  QuantumCircuit‑backed Hybrid in the QML code.

The design follows the same scaling logic as the two seed modules:
* image CNN part (QCNet‑style) – 1‑2 layers, 2‑3× down‑sampling.
* graph GNN part – 1‑2 message‑passing layers with `torch.nn.functional.elu`.
* combined head – linear + HybridFunction.

The class is fully compatible with the anchor path
`ClassicalQuantumBinaryClassification__gen153.py` and can be imported
directly as ``HybridQuantumGraphNet``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# --------------------------------------------------------------------------- #
# 1. Differentiable hybrid head (mimics quantum expectation)
# --------------------------------------------------------------------------- #
class HybridFunction(nn.Module):
    """Differentiable sigmoid‑like head that can be used interchangeably
    with the original quantum circuit.  The forward pass is a simple
    sigmoid, but the backward pass is left as the default autograd
    implementation so that the gradient flows through the preceding
    classical layers without any custom logic.  The class is kept
    lightweight to allow easy replacement by the quantum version
    in the QML code."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

# --------------------------------------------------------------------------- #
# 2. Classical graph neural network
# --------------------------------------------------------------------------- #
class GNNLayer(nn.Module):
    """Single message‑passing layer (linear + ELU)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, in_dim]
        # adj: [num_nodes, num_nodes]
        h = torch.matmul(adj, x)  # message aggregation
        h = torch.matmul(h, self.weight.t())
        return F.elu(h)

class GNN(nn.Module):
    """Two‑layer GNN with ELU activations."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = GNNLayer(in_dim, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h

# --------------------------------------------------------------------------- #
# 3. Hybrid head (classical linear + HybridFunction)
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a linear head
    followed by the differentiable HybridFunction."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.head = HybridFunction()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = self.head(logits)
        return probs

# --------------------------------------------------------------------------- #
# 4. CNN part (QCNet‑style)
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """CNN-based feature extractor mirroring the original quantum model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --------------------------------------------------------------------------- #
# 5. Unified hybrid graph‑image network
# --------------------------------------------------------------------------- #
class HybridQuantumGraphNet(nn.Module):
    """Combines CNN, GNN and a hybrid head into a single module."""
    def __init__(self,
                 gnn_in_dim: int,
                 gnn_hidden_dim: int,
                 gnn_out_dim: int):
        super().__init__()
        self.cnn = QCNet()
        self.gnn = GNN(gnn_in_dim, gnn_hidden_dim, gnn_out_dim)
        # hybrid head receives concatenated CNN output (1 dim) and
        # aggregated graph embedding (gnn_out_dim dims)
        self.hybrid = Hybrid(1 + gnn_out_dim)
    def forward(self,
                image: torch.Tensor,
                graph_features: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Image tensor of shape (B, 3, H, W).  B is typically 1.
        graph_features : torch.Tensor
            Node feature matrix of shape (N, gnn_in_dim).
        adj : torch.Tensor
            Adjacency matrix of shape (N, N).
        Returns
        -------
        torch.Tensor
            Binary probability vector of shape (B, 2).
        """
        # CNN branch
        cnn_out = self.cnn(image)          # (B, 1)
        # GNN branch
        gnn_out = self.gnn(graph_features, adj)   # (N, gnn_out_dim)
        # Aggregate graph representation (mean over nodes)
        gnn_mean = gnn_out.mean(dim=0, keepdim=True)  # (1, gnn_out_dim)
        # Broadcast to batch size
        gnn_mean = gnn_mean.expand(cnn_out.size(0), -1)  # (B, gnn_out_dim)
        # Concatenate
        combined = torch.cat([cnn_out, gnn_mean], dim=1)  # (B, 1+gnn_out_dim)
        # Hybrid head
        probs = self.hybrid(combined)  # (B, 1)
        # Convert to two‑class probabilities
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = [
    "HybridFunction",
    "GNNLayer",
    "GNN",
    "Hybrid",
    "QCNet",
    "HybridQuantumGraphNet",
]
