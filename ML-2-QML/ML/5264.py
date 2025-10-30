"""Hybrid kernel, transformer, graph, classifier module.

This module combines radial basis function kernels, quantum kernel simulation,
transformer blocks, graph-based state fidelity, and a feed‑forward classifier.
It is a fully classical implementation that mirrors the quantum counterparts
in the QML seed. The class is designed to be interchangeable with the
quantum version while providing a robust classical baseline.
"""

import torch
import torch.nn as nn
import numpy as np
import itertools
import networkx as nx
from typing import Sequence, Iterable, Tuple


# --------------------------------------------------------------------------- #
#   Kernel implementations
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalQuantumKernel(nn.Module):
    """Classical simulation of a quantum kernel via a random unitary."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        dim = 2 ** n_wires
        # Random unitary (real part only for simplicity)
        q, _ = np.linalg.qr(np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
        self.unitary = torch.from_numpy(q).float()  # shape (dim, dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode inputs as computational basis states
        ix = torch.argmax(x, dim=-1)
        iy = torch.argmax(y, dim=-1)
        state_x = torch.eye(2 ** self.n_wires)[ix]
        state_y = torch.eye(2 ** self.n_wires)[iy]
        # Apply unitary
        state_x = state_x @ self.unitary
        state_y = state_y @ self.unitary
        # Overlap
        overlap = torch.abs(torch.sum(state_x * state_y, dim=-1, keepdim=True))
        return overlap


# --------------------------------------------------------------------------- #
#   Transformer block (classical)
# --------------------------------------------------------------------------- #

class TransformerBlockClassical(nn.Module):
    """Single classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#   Graph utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Absolute squared overlap between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#   Hybrid model
# --------------------------------------------------------------------------- #

class HybridKernelClassifier(nn.Module):
    """
    Hybrid model that combines classical/quantum kernels, transformer blocks,
    graph-based state fidelity, and a feed‑forward classifier.
    """

    def __init__(self,
                 kernel_type: str = 'rbf',
                 gamma: float = 1.0,
                 n_wires: int = 4,
                 transformer_cfg: dict | None = None,
                 graph_threshold: float = 0.9,
                 classifier_type: str = 'classical',
                 classifier_depth: int = 3,
                 num_features: int | None = None) -> None:
        super().__init__()
        self.kernel_type = kernel_type
        self.graph_threshold = graph_threshold

        # Kernel
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(gamma)
        else:
            self.kernel = ClassicalQuantumKernel(n_wires)

        # Transformer
        if transformer_cfg is None:
            transformer_cfg = dict(embed_dim=64, num_heads=4, ffn_dim=128, dropout=0.1)
        self.transformer = TransformerBlockClassical(**transformer_cfg)

        # Classifier
        embed_dim = transformer_cfg['embed_dim']
        if classifier_type == 'classical':
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            # Quantum classifier simulated classically via a simple feed‑forward
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        self.num_features = num_features or embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, features).

        Returns
        -------
        torch.Tensor
            Logits for binary classification.
        """
        # Compute kernel matrix between input and itself
        batch = x.shape[0]
        kernel_matrix = torch.zeros(batch, batch, device=x.device)
        for i in range(batch):
            for j in range(batch):
                kernel_matrix[i, j] = self.kernel(x[i].unsqueeze(0), x[j].unsqueeze(0))

        # Treat kernel as sequence of embeddings
        embed_dim = self.transformer.norm1.normalized_shape[0]
        seq = kernel_matrix.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Transformer encoding
        h = self.transformer(seq)

        # Graph adjacency from state fidelity
        states = [h[i].mean(dim=0) for i in range(batch)]
        _ = fidelity_adjacency(states, self.graph_threshold)

        # For demonstration, use mean‑pooled representation
        pooled = h.mean(dim=1)  # shape (batch, embed_dim)
        logits = self.classifier(pooled)
        return logits


__all__ = ["HybridKernelClassifier", "RBFKernel", "ClassicalQuantumKernel",
           "TransformerBlockClassical", "state_fidelity", "fidelity_adjacency"]
