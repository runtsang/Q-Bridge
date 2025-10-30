"""
QuantumNATGraphHybrid: Classical + Quantum hybrid for image classification and graph analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import networkx as nx

# Import the quantum circuit from the QML module
from QuantumNATGraphHybrid_qml import QuantumCircuit

class QuantumNATGraphHybrid(nn.Module):
    """Hybrid model combining a classical CNN backbone, a Pennylane quantum layer,
    and a fidelity‑based adjacency graph for downstream tasks."""
    def __init__(self,
                 in_channels: int = 1,
                 cnn_features: int = 16,
                 n_qubits: int = 4,
                 graph_threshold: float = 0.95,
                 seed: int | None = None) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, cnn_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.quantum = QuantumCircuit(n_qubits=n_qubits, seed=seed)
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns the adjacency matrix of the fidelity graph
        built from the quantum states of the batch."""
        # Classical feature extraction
        features = self.flatten(self.cnn(x))  # shape (B, 16*7*7)
        # Encode each feature vector into a quantum state
        q_states_list = []
        for f in features:
            q_state = self.quantum(f.cpu().numpy())
            q_states_list.append(q_state)
        q_states = np.stack(q_states_list)  # shape (B, 2**n_qubits)
        # Build fidelity graph among batch states
        graph = self._build_fidelity_graph(q_states)
        # Return adjacency matrix as a tensor
        adj = nx.to_numpy_array(graph)
        return torch.tensor(adj, dtype=torch.float32)

    def _build_fidelity_graph(self, states: np.ndarray) -> nx.Graph:
        """Build a graph where nodes are batch samples and edges represent
        state fidelity above a threshold."""
        n = states.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                fid = np.abs(np.vdot(states[i], states[j])) ** 2
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph
