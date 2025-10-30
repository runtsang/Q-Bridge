"""Quantum hybrid model that encodes CNN-derived features into qubits,
applies a graph-based quantum layer, measures, and classifies.
It extends the original QuantumNAT quantum module by integrating
graph-based interactions among qubits derived from the hidden
representation."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools
import numpy as np

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared cosine similarity between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float, *,
                       secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def compute_adjacency_batch(features: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return an adjacency matrix (0/1) from pairwise state fidelities across batch."""
    norm = torch.norm(features, dim=1, keepdim=True) + 1e-12
    norm_feat = features / norm
    fidelity_matrix = torch.matmul(norm_feat, norm_feat.t())
    adjacency = (fidelity_matrix >= threshold).float()
    return adjacency

def compute_qubit_adjacency(hidden: torch.Tensor, threshold: float) -> torch.Tensor:
    """Adjacency between qubits based on similarity of hidden feature values."""
    diff = torch.abs(hidden.unsqueeze(0) - hidden.unsqueeze(1))  # (dim, dim)
    adjacency = (diff <= threshold).float()
    adjacency.fill_diagonal_(0.0)
    return adjacency

class QLayer(tq.QuantumModule):
    """Quantum graph layer that applies a random circuit and controlled rotations
    between qubits connected in the adjacency graph."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_qubits)))

    def forward(self, qdev: tq.QuantumDevice, adjacency: torch.Tensor):
        self.random_layer(qdev)
        # Controlled rotations between neighboring qubits
        for i in range(adjacency.size(0)):
            for j in range(adjacency.size(1)):
                if adjacency[i, j] > 0 and i < j:
                    tq.CNOT(qdev, wires=[i, j])
        return

class HybridNATModel(tq.QuantumModule):
    """Quantum hybrid model that encodes CNN-derived features into qubits,
    applies a graph-based quantum layer, measures, and classifies."""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 hidden_dim: int = 32,
                 threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.n_qubits = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, hidden_dim)
        self.q_layer = QLayer(self.n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc2 = nn.Linear(self.n_qubits, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (bsz, 16, 7, 7)
        flattened = features.view(features.size(0), -1)  # (bsz, 16*7*7)
        hidden = torch.relu(self.fc1(flattened))  # (bsz, hidden_dim)

        logits_list = []
        for i in range(hidden.size(0)):
            feat = hidden[i]  # (hidden_dim,)
            adjacency_qubits = compute_qubit_adjacency(feat, threshold=0.5)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device=x.device, record_op=True)
            # Encode each feature into a rotation on its qubit
            for j, val in enumerate(feat):
                tq.RX(qdev, params=val, wires=j)
            self.q_layer(qdev, adjacency_qubits)
            out = self.measure(qdev)  # (1, n_qubits)
            out = self.fc2(out)  # (1, num_classes)
            logits_list.append(out)
        logits = torch.cat(logits_list, dim=0)  # (bsz, num_classes)
        return self.norm(logits)

__all__ = ["HybridNATModel"]
