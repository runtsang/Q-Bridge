import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
from typing import List, Tuple

class HybridQuantumNAT(nn.Module):
    """
    Hybrid classical‑quantum model that fuses a CNN encoder with a graph‑based quantum layer.
    The encoder extracts spatial features; the quantum layer models relationships via a
    fidelity‑based adjacency graph and a parameterized circuit implemented with torchquantum.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, n_wires: int = 4):
        super().__init__()
        # Classical CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Feature size after pooling: 16 * 7 * 7 (for 28x28 input)
        self.feature_dim = 16 * 7 * 7
        # Quantum layer parameters
        self.n_wires = n_wires
        self.encoder_q = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        # Trainable parameterized gates per wire
        self.param_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_wires)) for _ in range(self.n_wires)]
        )
        # Fully connected head after quantum measurement
        self.head = nn.Sequential(
            nn.Linear(self.n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def _build_fidelity_graph(self, features: torch.Tensor) -> nx.Graph:
        """
        Build a graph where nodes are feature vectors and edges are added if cosine similarity
        exceeds a threshold.  This mirrors the GraphQNN fidelity_adjacency.
        """
        norms = features.norm(dim=1, keepdim=True)
        sims = (features @ features.t()) / (norms @ norms.t() + 1e-12)
        g = nx.Graph()
        g.add_nodes_from(range(features.size(0)))
        threshold = 0.9
        for i in range(features.size(0)):
            for j in range(i + 1, features.size(0)):
                if sims[i, j] >= threshold:
                    g.add_edge(i, j, weight=1.0)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features, build graph, run quantum circuit, and classify.
        """
        batch_size = x.size(0)
        # Encode classical features
        feat = self.encoder(x)  # shape (B, 16, 7, 7)
        feat_flat = feat.view(batch_size, -1)  # (B, feature_dim)
        # Build fidelity graph over batch items
        graph = self._build_fidelity_graph(feat_flat)
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device, record_op=True)
        # Encode features into qubits
        self.encoder_q(qdev, feat_flat)
        # Apply random layer
        self.random_layer(qdev)
        # Apply parameterized gates per wire
        for i, gate in enumerate(self.param_gates):
            tqf.rx(qdev, gate, wires=i)
        # Measure all qubits
        out = tq.MeasureAll(tq.PauliZ)(qdev)
        # Pass through head
        logits = self.head(out)
        return self.norm(logits)
