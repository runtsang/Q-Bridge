"""QuantumHybridNet – classical hybrid architecture.

This module implements a hybrid neural network that fuses:
- a CNN backbone (from the QuantumNAT ML seed)
- an optional classical LSTM (or QLSTM in the quantum variant)
- a simulated “quantum” layer (dense mapping)
- a regression head
- graph‑based feature aggregation

The class is fully PyTorch‑compatible and can be used as a drop‑in replacement for the original QFCModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# 1. CNN backbone – simple 2‑layer conv network
# --------------------------------------------------------------------------- #
class _CNNBase(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 2. Simulated “quantum” layer – dense mapping
# --------------------------------------------------------------------------- #
class _SimulatedQuantumLayer(nn.Module):
    """Replace a variational quantum circuit with a trainable dense layer."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

# --------------------------------------------------------------------------- #
# 3. Graph‑based feature aggregation
# --------------------------------------------------------------------------- #
def _fidelity_graph(
    features: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph where edge weight reflects cosine similarity."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(features)))
    for i, fi in enumerate(features):
        for j in range(i + 1, len(features)):
            fj = features[j]
            cos_sim = F.cosine_similarity(fi.unsqueeze(0), fj.unsqueeze(0)).item()
            if cos_sim >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and cos_sim >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 4. Hybrid network
# --------------------------------------------------------------------------- #
class QuantumHybridNet(nn.Module):
    """
    Classical hybrid model that fuses a CNN backbone, an optional LSTM,
    a simulated quantum layer, and a regression head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_qubits: int = 4,
        lstm_n_qubits: int = 0,
        hidden_dim: int = 32,
        output_dim: int = 1,
        graph_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.cnn = _CNNBase(in_channels, out_features=4)
        self.lstm_n_qubits = lstm_n_qubits
        if lstm_n_qubits > 0:
            # Placeholder: quantum LSTM will be provided in the quantum variant
            self.lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        self.quantum_layer = _SimulatedQuantumLayer(4, n_qubits)
        self.regression_head = nn.Sequential(
            nn.Linear(n_qubits + hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image batch.
        """
        cnn_feat = self.cnn(x)          # shape (B, 4)
        lstm_out, _ = self.lstm(cnn_feat.unsqueeze(1))
        lstm_feat = lstm_out.squeeze(1)  # shape (B, hidden_dim)
        qfeat = self.quantum_layer(cnn_feat)  # shape (B, n_qubits)
        combined = torch.cat([lstm_feat, qfeat], dim=1)
        out = self.regression_head(combined)
        return out

    def build_fidelity_graph(self, features: torch.Tensor, threshold: float | None = None) -> nx.Graph:
        """
        Build a graph from a batch of feature vectors using cosine similarity.
        """
        thresh = threshold if threshold is not None else self.graph_threshold
        return _fidelity_graph(features, thresh)

__all__ = ["QuantumHybridNet"]
