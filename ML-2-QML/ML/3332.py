"""Hybrid classical pipeline: classical convolution → quantum patch encoder → graph aggregation."""

from __future__ import annotations

import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Classical patch extractor
# --------------------------------------------------------------------------- #
class ClassicalPatchExtractor(nn.Module):
    """
    A lightweight 2×2 patch extractor that mirrors the stride‑2 2‑D convolution
    used in the original QuanvolutionFilter.  It produces a flattened
    feature vector of shape (batch, 4) for each 28×28 image.
    """
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 4 output channels (2×2 patches)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (B, 4, 196)."""
        features = self.conv(x)                     # (B, 4, 14, 14)
        return features.view(x.size(0), 4, -1)      # (B, 4, 196)

# --------------------------------------------------------------------------- #
# Quantum patch encoder
# --------------------------------------------------------------------------- #
class QuantumPatchEncoder(tq.QuantumModule):
    """
    Variational quantum kernel applied to each 2×2 patch.  The encoder
    consists of a 4‑qubit block that maps the 4 pixel intensities to
    a 4‑dimensional measurement vector.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=4 * n_layers, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, 196) – 2×2 patches flattened.
        Returns: (B, 4, 196) – measurement on 4 qubits per patch.
        """
        bsz, _, n_patches = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # reshape patches into (B, 196, 4) so each patch is a 2‑bit
        patches = x.permute(0, 2, 1).contiguous()   # (B, 196, 4)
        outputs = []
        for idx in range(n_patches):
            data = patches[:, idx, :]  # (B, 4)
            self.encoder(qdev, data)
            self.random_layer(qdev)
            measurement = self.measure(qdev)
            outputs.append(measurement.view(bsz, 4))
        return torch.cat(outputs, dim=1)        # (B, 4*196)

# --------------------------------------------------------------------------- #
# Graph‑based aggregation
# --------------------------------------------------------------------------- #
class GraphAggregation(nn.Module):
    """
    Builds a weighted adjacency graph from the quantum‑encoded patches
    using fidelity‑based edge weights and then performs a simple mean
    pooling over the graph nodes.
    """
    def __init__(self, threshold: float = 0.8, secondary: float | None = None,
                 secondary_weight: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def _state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Return the squared overlap between two 4‑D vectors."""
        an = a / (torch.norm(a) + 1e-12)
        bn = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(an, bn).item() ** 2)

    def forward(self, quantum_features: torch.Tensor) -> torch.Tensor:
        """
        quantum_features: (B, 4*196) – flattened quantum output from QP‑encoder.
        """
        B, N = quantum_features.shape
        out = []
        for i in range(B):
            vecs = quantum_features[i].view(14 * 14, 4)  # (196, 4)
            graph = nx.Graph()
            graph.add_nodes_from(range(196))
            for (u, v) in itertools.combinations(range(196), 2):
                fid = self._state_fidelity(vecs[u], vecs[v])
                if fid >= self.threshold:
                    graph.add_edge(u, v, weight=1.0)
                elif self.secondary and fid >= self.secondary:
                    graph.add_edge(u, v, weight=self.secondary_weight)
            node_feats = torch.stack([vecs[n] for n in graph.nodes], dim=0)  # (196, 4)
            pooled = torch.mean(node_feats, dim=0)  # (4,)
            out.append(pooled)
        return torch.stack(out)  # (B, 4)

# --------------------------------------------------------------------------- #
# Full hybrid model
# --------------------------------------------------------------------------- #
class QuanvolutionHybridGraphQL(nn.Module):
    """
    End‑to‑end model that chains the ClassicalPatchExtractor, QuantumPatchEncoder,
    GraphAggregation and a final linear classifier.
    """
    def __init__(self, num_classes: int = 10, threshold: float = 0.8,
                 secondary: float | None = None, hidden_dim: int = 32) -> None:
        super().__init__()
        self.extractor = ClassicalPatchExtractor()
        self.quantum = QuantumPatchEncoder()
        self.graph = GraphAggregation(threshold=threshold, secondary=secondary)
        self.classifier = nn.Linear(4, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)                   # (B, 4, 196)
        quantum = self.quantum(x)                      # (B, 4*196)
        graph_feat = self.graph(quantum)               # (B, 4)
        logits = self.classifier(graph_feat)           # (B, hidden_dim)
        return self.head(logits)                       # (B, num_classes)

__all__ = ["QuanvolutionHybridGraphQL"]
