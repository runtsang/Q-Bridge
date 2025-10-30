"""Hybrid CNN + quantum head for binary classification with optional graph features.

The model extends the classical CNN used in the original seed, but replaces the
final dense layer with a HybridQuantumLayer that maps a vector of parameters
to a quantum expectation value.  A lightweight graph encoder, built on the
GraphQNN utilities from the quantum module, can be concatenated with the
image representation before the quantum head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import qutip as qt

# Import quantum utilities
from qml_code import HybridQuantumLayer as _HybridQuantumLayer
from qml_code import feedforward, random_network

class HybridQuantumLayer(_HybridQuantumLayer):
    """Alias for the quantum layer used in the hybrid model."""
    pass

class HybridCNNGraphClassifier(nn.Module):
    """CNN backbone followed by a quantum expectation head.
    Optionally accepts a networkx graph, whose embedding is produced
    by a lightweight graph‑QNN and concatenated with the image
    representation before the quantum layer."""

    def __init__(self,
                 image_in_channels: int = 3,
                 image_size: tuple[int, int] = (32, 32),
                 graph_arch: list[int] | None = None,
                 graph_shots: int = 100,
                 quantum_shifts: float = 0.0):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(image_in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Compute flattened feature size
        dummy = torch.zeros(1, image_in_channels, *image_size)
        x = self._forward_conv(dummy)
        flat_size = x.shape[1]

        # Fully‑connected layers
        self.fc1 = nn.Linear(flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum head
        self.quantum = HybridQuantumLayer(n_qubits=1, shots=100, shift=quantum_shifts)

        # Optional graph encoder
        self.graph_arch = graph_arch
        if graph_arch is not None:
            _, unitaries, _, _ = random_network(graph_arch, samples=1)
            self.graph_unitaries = unitaries
        else:
            self.graph_unitaries = None

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def _graph_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """Generate a quantum‑state embedding for a graph using the
        GraphQNN utilities.  The embedding is the final state vector
        produced by the random network defined in the quantum module.
        """
        # Construct a pure state from the degree distribution
        degrees = np.array([d for _, d in graph.degree()], dtype=np.float32)
        if degrees.size == 0:
            degrees = np.array([1.0], dtype=np.float32)
        degrees = degrees / (degrees.sum() + 1e-8)

        # Pad or truncate to match the input size of the network
        target_len = self.graph_arch[0]
        if degrees.size < target_len:
            degrees = np.pad(degrees, (0, target_len - degrees.size), mode="constant")
        elif degrees.size > target_len:
            degrees = degrees[:target_len]

        state = qt.Qobj(degrees.reshape(-1, 1))
        samples = [(state, None)]
        states = feedforward(self.graph_arch, self.graph_unitaries, samples)
        embedding = states[0][-1].full().flatten()
        return torch.tensor(embedding, dtype=torch.float32, device="cpu")

    def forward(self, image: torch.Tensor,
                graph: nx.Graph | None = None) -> torch.Tensor:
        x = self._forward_conv(image)
        if self.graph_unitaries is not None and graph is not None:
            g_emb = self._graph_embedding(graph).to(x.device)
            x = torch.cat([x, g_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.quantum(x).unsqueeze(1)
        return torch.cat([probs, 1 - probs], dim=-1)

class EstimatorQNNRegressor(nn.Module):
    """Simple regression head that optionally ends with a quantum layer,
    mirroring the EstimatorQNN example."""
    def __init__(self, input_dim: int, hidden_dim: int = 8,
                 use_quantum: bool = True, shift: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.use_quantum = use_quantum
        if use_quantum:
            self.quantum = HybridQuantumLayer(n_qubits=1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.net(inputs)
        if self.use_quantum:
            out = self.quantum(out)
        return out

__all__ = [
    "HybridQuantumLayer",
    "HybridCNNGraphClassifier",
    "EstimatorQNNRegressor",
]
