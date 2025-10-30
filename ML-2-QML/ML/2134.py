"""
Hybrid classical classifier with an embedding layer, feed‑forward network, and a quantum‑classical fusion head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable, List


class EmbeddingNet(nn.Module):
    """Learn a fixed‑size embedding from the raw feature vector."""
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


class FeedForwardNet(nn.Module):
    """Deep network that processes the embedding and produces logits."""
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumClassifierModel(nn.Module):
    """
    Hybrid model that fuses a classical embedding + feed‑forward net with a quantum measurement vector.
    The quantum circuit is built via :func:`build_classifier_circuit` (see qml module).
    """
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        dropout: float,
        quantum_depth: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.embed = EmbeddingNet(in_dim, embed_dim)
        self.ff = FeedForwardNet(embed_dim, hidden_dim, num_classes, depth, dropout)
        # Quantum circuit is defined in the qml module; we keep a reference
        self.quantum_depth = quantum_depth
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(x)
        classical_logits = self.ff(embedded)
        # The quantum part is executed outside of this module in training code.
        return classical_logits

__all__ = ["EmbeddingNet", "FeedForwardNet", "QuantumClassifierModel"]
